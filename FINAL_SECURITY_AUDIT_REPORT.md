# MarketPulse AI: Security Audit Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22

**Supersedes:** the security report of `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`, whose verdict was "CONDITIONAL PASS". Its five open conditions were no rate limiting, root containers, no dashboard authentication, the in-memory registry with multiple workers, and no TLS guidance. All five are addressed below.

## Evidence labels

- **[LIVE]**: verified against a running server over HTTP or in Chrome, in this release cycle, on the final build unless noted.
- **[TEST]**: covered by automated tests in the final full-suite run.
- **[STATIC]**: code or configuration inspection.
- **[NOT VERIFIED]**: not checked.

## Verdict

**PASS for a single-instance deployment behind the provided HTTPS proxy, in production mode.** That configuration requires:

- `MARKETPULSE_ENV=production`
- `MARKETPULSE_API_KEY`
- `MARKETPULSE_DASHBOARD_PASSWORD_HASH`
- `CORS_ORIGINS` set to your domain(s)

In production the app refuses to run with the API key or the dashboard hash missing. The remaining limitations (§7) are design boundaries, not open vulnerabilities.

## 1. Controls added in this release

| Control | Behaviour | Evidence |
|---|---|---|
| **API rate limiting** | Sliding 60-second window per client IP:<br>• upload 10/min<br>• compute 30/min (optimize, simulate, ask, confirm, decisions run)<br>• everything else 120/min<br><br>`/` and `/health` are exempt. Over the limit, the API returns HTTP 429 `{"error":"rate_limited",…}` with `Retry-After`. It is on by default in production and off by default in development; `MARKETPULSE_RATE_LIMIT=on/off` overrides, and the per-class limits can be configured. Tracked clients are capped at 10,000 (LRU). | [LIVE] 11th upload in a minute → 429, `Retry-After: 60`; [TEST] 6 tests |
| **Dashboard authentication** | Single team password, stored only as `pbkdf2_sha256$600000$salt$hash` in the environment or `st.secrets`. Comparison uses `hmac.compare_digest`. Five failures lock sign-in for 60 s. There is no plaintext-password setting. In production, a missing hash makes the dashboard refuse to serve data. Generate the hash with `python -m dashboard.components.auth` (minimum 12 characters). | [LIVE] Chrome against the Docker container: 0 nav items before sign-in; wrong password rejected; 18 nav items after sign-in; [TEST] 8 tests |
| **Non-root containers** | The entrypoint runs as root only to fix volume ownership, then drops to the owner of `./data` (if non-root) or to uid/gid 10001 (`setgroups`/`setgid`/`setuid`) before starting the app. | [LIVE] PID 1 uid 10001 in both containers; [TEST] real privilege-drop tests |
| **Single worker** | The Dockerfile runs `--workers 1`. Production raises `RuntimeError` if `WEB_CONCURRENCY` > 1; development warns. | [TEST] 3 tests |
| **HTTPS** | `deploy/Caddyfile` (automatic certificates, HSTS, nosniff, frame-deny, body limits), `deploy/nginx.conf` (TLS, WebSocket, `limit_req`), and a Compose overlay that unpublishes the app ports. | [STATIC]; overlay validated with `docker compose config` [LIVE]; certificate issuance [NOT VERIFIED] |
| **Forwarded client IP** | The uvicorn API runs with `--proxy-headers`. `FORWARDED_ALLOW_IPS="*"` is set only in the proxy overlay, where the API port is not published; the README explains why. | [STATIC] |
| **Dependency audit** | pip-audit found 0 known vulnerabilities in 60 packages. The unused `anthropic` package was removed. | [LIVE] see `FINAL_DEPENDENCY_SECURITY_REPORT.md` |

## 2. API authentication (from the previous release, re-checked)

| Check | Result | Evidence |
|---|---|---|
| Production without `MARKETPULSE_API_KEY` | Refuses to start (exit 1, `RuntimeError`), under uvicorn and in Docker | [LIVE] final build |
| Protected endpoints without a key or with a wrong key | 401. The key is compared with `hmac.compare_digest`. | [LIVE] `/ask`; [TEST] all 8 protected endpoints |
| Valid key | 200 | [LIVE] |
| Key in logs | 0 occurrences (uvicorn and container logs) | [LIVE] |
| Invalid `MARKETPULSE_ENV` value | Refuses to start | [TEST] |
| Fail-closed if production has no key at request time | 503 | [TEST] |

Protected endpoints:

- `POST /upload`
- `POST /datasets/{id}/confirm`
- `POST /optimize-budget`
- `POST /simulate-budget`
- `POST /ask`
- `GET /datasets/{id}/decisions`
- `POST /datasets/{id}/decisions/run`
- `GET /kpis?dataset_id=<uploaded>`

Public endpoints expose only the bundled synthetic demo model and its outputs, never uploaded data.

## 3. Input handling

| Control | Evidence |
|---|---|
| Upload filename reduced to a safe basename, with a containment check (path traversal fixed in the previous release) | [TEST] 11 filename cases |
| `dataset_id` checked against a strict regex | [TEST] |
| Only `.csv` and `.xlsx` accepted; size limit 50 MB (413 above it); row cap 100,000 (configurable) | [TEST] |
| Optimizer: infeasible bounds or a non-positive budget return **400** with a reason. Previously the API answered `success` with violated constraints. | [TEST] |
| Dashboard upload: friendly errors for empty files, no header row, broken structure, unknown encoding or an invalid workbook. No paths or tracebacks shown. | [STATIC] [TEST] |
| User text rendered in HTML is escaped (`md_inline` escapes before formatting; KPI labels escape `<` and `>`) | [TEST] |
| Excel formula injection sanitised in exports (existing) | [TEST] |
| Pickle: only the bundled model, loaded from a fixed path | [STATIC] |

## 4. Secrets and configuration (scan of the staged release tree) [LIVE]

| Check | Result |
|---|---|
| `.env`, `secrets.toml` | None; only `.env.example`, whose secret values are empty ([TEST] `test_env_example_has_no_values_for_secrets`) |
| Key and token patterns (Anthropic, AWS, GitHub, Slack, private keys, PBKDF2 hashes) | None found |
| Hardcoded local paths in source or config | None. The only hit was in the previous test log, which is replaced in this release. |
| `.dockerignore` | Excludes `.env`, `.streamlit/secrets.toml`, `data/uploads/*` and caches ([TEST]) |
| Docker `.env` with `$` in the hash | Must be written as `$$`. Documented, and the hash was verified intact inside the container. [LIVE] |

## 5. CORS

`CORS_ORIGINS` unset means localhost development origins only, never `*`. Production logs a warning if it is unset or contains `*` (previous release, live-tested; unchanged code). [STATIC]

## 6. Threat notes

- **Brute force on the dashboard password.** The per-session lockout (5 tries / 60 s) slows one browser session but not an attacker who opens new sessions. Use a long random password (the hash generator refuses fewer than 12 characters) and the proxy's rate limiting (`limit_req` in `deploy/nginx.conf`).
- **The rate limiter is per process.** It resets on restart; this is acceptable for the single-instance design.

## 7. Remaining limitations

1. One shared dashboard password: no per-user accounts, roles or user audit log.
2. In-process state (upload registry and rate limiter) prevents horizontal scaling without adding a shared store.
3. Only the API has a built-in rate limiter; the dashboard relies on the proxy.
4. The official `python:3.13-slim` base image was **[NOT VERIFIED]**, because Docker Hub is unreachable here; a local substitute was used.
5. Real HTTPS certificate issuance was **[NOT VERIFIED]**.
6. OSV vulnerability data was **[NOT VERIFIED]** (blocked); the PyPI service was used.
7. No penetration test or external review was performed. **No independent third-party acceptance is claimed.**
