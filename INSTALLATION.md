# Installation and configuration

Verified on **Python 3.13.18 on Linux** (tested: passed). **Windows: not yet verified on a Windows machine.** Use
`WINDOWS_SETUP_GUIDE.md`, which has PowerShell scripts and a one-command validator. Docker: see `DOCKER_SETUP_GUIDE.md`.
`requirements.txt` declares wider version ranges, but only the versions in `FINAL_TEST_EXECUTION_LOG.txt` were tested.

## 1. Get the code

```bash
unzip MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip
cd MarketPulse_AI_FINAL_RELEASE
sha256sum ../MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip   # compare with SHA256_CHECKSUM.txt
```

## 2. Create an environment and install

```bash
python3.13 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt       # required
pip install -r requirements-optional.txt   # optional: XGBoost, LightGBM, CatBoost
```

The optional libraries only add three models to the Predictive tab. Without them those
three models appear as *unavailable* with the reason, and everything else works.
LightGBM needs the OpenMP runtime: `sudo apt-get install libgomp1` on Debian or Ubuntu,
or `brew install libomp` on macOS. The macOS command was not verified.

## 3. Configure (optional for local use)

```bash
cp .env.example .env
```

The full table of variables is in `README.md` → *Environment Setup*. The ones that matter most:

| Variable | Default | When to set it |
|---|---|---|
| `MARKETPULSE_ENV` | `development` | Set `production` for any shared deployment |
| `MARKETPULSE_API_KEY` | empty | Required in production; sent as `X-API-Key` |
| `MARKETPULSE_DASHBOARD_PASSWORD_HASH` | empty | Required in production (`python -m dashboard.components.auth`) |
| `CORS_ORIGINS` | localhost | Your front-end origin(s); never `*` |
| `MARKETPULSE_MAX_UPLOAD_MB` / `MARKETPULSE_MAX_ROWS` | 50 / 100000 | API upload limits (rows beyond the limit are dropped and reported in the response) |
| `INSTALL_OPTIONAL_ML` | false | Docker only: build the dashboard image with the optional models |

The dashboard's upload limit is Streamlit's `maxUploadSize` in `.streamlit/config.toml`
(200 MB).

## 4. Run

```bash
streamlit run app.py                       # dashboard → http://localhost:8501
uvicorn api.main:app --port 8000           # API → http://127.0.0.1:8000/docs (docs only in development)
```

The demo dataset (NovaMart, synthetic, INR) and its trained model ship with the project.

## 5. Test

```bash
python -I -m pytest tests/ -q
```

The expected result is in `FINAL_TEST_REPORT.md`. The user's real Kaggle file is included at
`tests/fixtures/part02/real/synthetic_mmm_weekly_india_SAT.csv`, so its tests run. If you remove that file, those tests
are skipped with an explanation instead.

## 6. Docker

```bash
cp .env.example .env    # set MARKETPULSE_ENV=production, MARKETPULSE_API_KEY, password hash
docker compose up --build -d
```

Set `INSTALL_OPTIONAL_ML=true` in `.env` to include XGBoost, LightGBM and CatBoost in the dashboard image. Full
instructions and verification status: `DOCKER_SETUP_GUIDE.md` and `DEPLOYMENT_READINESS_REPORT.md`.
