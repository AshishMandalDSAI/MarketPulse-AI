# Deployment readiness: PART 06 (final verified release)

**Nothing was deployed, pushed or published.** All checks ran locally in the build environment.

## 1. Verdict

| Area | Status |
|---|---|
| Local run from the release ZIP, clean virtual environment (Linux) | **Tested: passed** (`FINAL_TEST_REPORT.md`) |
| `docker compose` configuration (main + proxy override) | **Tested: passed** (`docker compose config -q`) |
| Build from the **official** `python:3.13-slim` | **Blocked.** On 2026-11-23, `docker pull python:3.13-slim` failed with *"Forbidden (agent proxy: no rule or network setting allows this host)"* for `registry-1.docker.io`, and `deb.debian.org` did not resolve. PyPI was reachable. |
| Build, run and the 10 required checks with a **local stand-in base image** | **Tested with limitations**: all checks pass (§3) |
| Optional ML (XGBoost, LightGBM, CatBoost) inside Docker | **Tested with limitations** on the stand-in base plus the PyPI wheels: 12/12 models available and run; fallback without `libgomp1` is graceful (§4) |
| Production readiness | **Not claimed.** The official-image build, a Windows run and an external penetration test are still outstanding (`KNOWN_LIMITATIONS.md` §1). |

## 2. What "stand-in base image" means

The image tagged `python:3.13-slim` in this environment is the locally built substitute documented in the baseline
(`FINAL_DEPLOYMENT_VERIFICATION.md`): Python 3.13.18, an **`apt-get` stub that skips installs**, and an offline wheel
directory (`PIP_NO_INDEX`). For the optional-ML check, a test-only derived base added the PyPI wheels of
`requirements-optional.txt` and the OpenMP runtime (`libgomp.so.1`), which emulates `apt-get install libgomp1`.

So these builds prove the project's Dockerfiles, entrypoint, compose file, health checks, privilege drop,
persistence and application behaviour. They do **not** prove that `apt-get` or a PyPI download succeed inside the
official image. **The release Dockerfiles are written for the official image**; the only Dockerfile change in PART 06
is the opt-in `INSTALL_OPTIONAL_ML` argument.

## 3. The 10 required checks: fresh extraction of the release, production `.env`

Two complete runs: **default image** (`INSTALL_OPTIONAL_ML=false`) and **optional-ML image** (`true`).

| # | Check | Default image | Optional-ML image |
|---|---|---|---|
| 1 | Image builds (`docker compose build --no-cache`, both services) | PASS | PASS |
| 2 | Containers start and become **healthy** | PASS | PASS |
| 3 | Health endpoints: API `/health` 200, dashboard `/_stcore/health` 200 | PASS | PASS |
| 4 | Dashboard loads: home page 200; Chrome sign-in, then all 16 pages with 0 exceptions, 0 error boxes and 0 console errors; a wrong password stays locked | PASS | PASS (Chrome pass run on this image) |
| 5 | API works: upload (key) → `dataset_id` → confirm → KPIs, all 200 | PASS | PASS |
| 6 | **Non-root:** uid 10001 in both containers | PASS | PASS |
| 7 | **File permissions:** host `data/uploads/<id>` owned by 10001, mode 755. As uid 10001, `data/` and `models/` are writable and the application code is **read-only** | PASS* | PASS |
| 8 | **Persistence:** the uploaded and confirmed dataset is still served after `restart`, **and** after `down` + `up` (new containers) | PASS | PASS |
| 9 | **Restart:** both healthy again after `docker compose restart` | PASS | PASS |
| 10 | **Security controls after restart:** no key gives 401; `..%2F..%2Fetc` as a dataset ID gives 400; a 60 MB upload gives 413; `/docs` gives 404 in production; `nosniff` and `DENY` headers present | PASS | PASS |
| + | Container logs contain no tracebacks, and the API key never appears in them | PASS | PASS |
| + | Optional ML in the container (as uid 10001) | 0 of 3 libraries available; **9 models ran**; XGBoost, LightGBM and CatBoost reported *unavailable* (expected) | **3 of 3 available; all 12 models ran**, each with its own holdout row |

\* In the first run, check 7 used `docker exec` as **root**, so it wrongly reported the code as writable. Re-run as
uid 10001: PASS on both containers. The check was corrected for the second run.

**Graceful fallback** was also tested in a container built with the optional packages but **without** `libgomp1`.
LightGBM was reported as *"installed but cannot be loaded (OSError: libgomp.so.1: cannot open shared object file)"*,
and the other 11 models ran.

## 4. To finish the verification (needs a machine with Docker Hub access)

Run `DOCKER_SETUP_GUIDE.md` §5. Expect the same results as §3, plus a real `apt-get install` of `build-essential` and
`libgomp1`.

---

# Earlier PART 05 report (kept for history)

# Deployment readiness report (upgrade PART 05)

**Nothing was deployed.** No image was pushed, no hosted app was updated and nothing was
published. Everything below ran locally in the build environment.

## 1. Verdict

| Area | Status |
|---|---|
| Local run (dashboard and API) from the release ZIP in a clean virtual environment | **Verified** (§2) |
| `docker compose` configuration (main file and the `deploy/` proxy override) | **Verified** (`docker compose config -q`) |
| Image build and run with the **official** `python:3.13-slim` | **NOT VERIFIED.** Docker Hub is blocked by this environment's network proxy |
| Image build and run with a **local substitute** of `python:3.13-slim` (the baseline's method) | **Verified** (§3), with the caveats below |
| Production hardening (API key, dashboard password, non-root, CORS, rate limits) | Configured by the baseline, unchanged, live-checked (§3) |
| Ready for a production deployment | **Not claimed.** See §5 |

## 2. Local run from the release ZIP

The release ZIP was extracted fresh and installed into a new virtual environment with
Python 3.13.18, `requirements.txt` and `requirements-optional.txt`. 71 packages were installed;
`part05/clean_env_freeze.txt` lists them in the build workspace. Results:

- Dashboard and API start in about 0.9 s each.
- The full test suite passes; see `FINAL_TEST_REPORT.md` for the counts.
- **Extraction as a non-root user:** the release ZIP extracts with directories at 0755 and
  `docker-entrypoint.sh` executable, and `NONROOT_EXTRACT_OK` was confirmed.

**Packaging defect found and fixed in PART 05.** The helper script that built the PART 01–04
backup ZIPs wrote directory entries with mode 0600. When a **non-root** user extracts those
backups with Info-ZIP `unzip`, the directories are not traversable. Root was unaffected, which
is why the earlier root-run tests passed. Docker was also affected: the bind-mounted `data/`
became unreadable after the entrypoint dropped to uid 10001, and the dashboard container
restarted in a loop. This was observed on the first PART 05 Docker attempt.

- **Fix:** the builder now writes directories as 0755, and the final ZIP was built with it.
- **Previous backups were not modified**, as instructed.
- **Workaround for those older ZIPs:** after extracting, run
  `chmod -R u+rwX,go+rX MarketPulse_AI_FINAL_RELEASE`. Extracting with Python's `zipfile`
  module is not affected.
- The baseline ZIP (`MarketPulse_AI_ULTIMATE_FINAL_RELEASE.zip`) was never affected.

## 3. Docker: what was actually run

**Base-image caveat.** A local image tagged `python:3.13-slim` (ID `65877e5e7b44`) already
existed in this environment's Docker daemon. It is the **locally built substitute** that the
baseline's `FINAL_DEPLOYMENT_VERIFICATION.md` documents. It contains:

- Python 3.13.18 on a minimal root filesystem;
- an `apt-get` stub that **skips** `apt-get update` and `install`;
- a local wheel directory used through `PIP_NO_INDEX` / `PIP_FIND_LINKS`, set in that
  substitute image only.

So the build below proves that the project's Dockerfiles, entrypoint, compose file, health checks,
privilege drop and application work. It does **not** prove that `apt-get install build-essential`
or a `pip install` from PyPI succeed inside the official image. **The release Dockerfiles are
unchanged.**

Steps on a fresh extract of the fixed ZIP, with a production `.env`
(`MARKETPULSE_ENV=production`, a random API key, and a PBKDF2 dashboard password hash with
`$` written as `$$`):

| Check | Result |
|---|---|
| `docker compose build --no-cache` | OK, 1 min 15 s (the base-image layer comes from the local substitute) |
| `docker compose up -d` | `marketpulse-dashboard` **healthy**, then `marketpulse-api` **healthy** |
| Dashboard `/_stcore/health` | 200 |
| API `/health` | 200, `pipeline_ready: true` |
| Process user in both containers | uid/gid **10001** (non-root) |
| `POST /upload` with no key / with a wrong key | **401 / 401** |
| `POST /upload` with the key (`spend_weekly.csv`) | 200, a dataset ID is returned |
| `POST /datasets/..%2F..%2Fetc/confirm` | 404 (not routed) |
| `GET /kpis?dataset_id=../../etc` | **400** "Invalid dataset_id … must match `^[a-zA-Z0-9_-]{1,64}$`" |
| Upload of 60,000,000 bytes (above 50 MiB) | **413** "File exceeds the 50MB upload limit." |
| Public `GET /kpis` (demo) | 200 |
| Chrome (Playwright) against the container | Sign-in form shown; a wrong password stays locked; the correct password unlocks; all 16 pages on the demo show 0 exceptions, 0 error boxes and 0 console errors |
| Optional ML libraries inside the image | Reported *unavailable* with an install hint. This is expected, because the images install only `requirements.txt` |
| Container logs | No `Traceback` or `ERROR` lines |

Raw outputs are in the build workspace: `part05/docker_build.log`, `docker_live_checks.txt`,
`docker_compose_logs.txt` and `docker_browser.json`.

## 4. Configuration checklist for a real deployment

1. Build against the **official** `python:3.13-slim` on a machine with Docker Hub access, then
   re-run §3's checks. (`deploy/README.md` §2 has the commands.)
2. In `.env`, set `MARKETPULSE_ENV=production`, a long random `MARKETPULSE_API_KEY`,
   `MARKETPULSE_DASHBOARD_PASSWORD_HASH` (with `$$`), and `CORS_ORIGINS` set to your real origin.
3. Put the services behind TLS using `deploy/docker-compose.proxy.yml` with Caddy or nginx.
4. Keep the API at a single worker.
5. If the optional models are needed in containers, add `requirements-optional.txt` and
   `libgomp1` to the Dockerfiles. This is not done or tested here.
6. Lower Streamlit's `server.maxUploadSize` if 200 MB is more than you want to accept.
7. Re-run `pip-audit` at deployment time.

## 5. Why "production-ready" is not claimed

- The official base image was not built.
- No load or concurrency test was run.
- No penetration test or container CVE scan was performed.
- Verification covered only Linux and Chrome.
- The real Kaggle file was not tested.

All the evidence above supports a **deployment candidate**, pending the checks in §4.
