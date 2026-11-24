# MarketPulse AI — Dependency Security Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22
Evidence labels: **[EXECUTED]** · **[STATIC]** · **[NOT VERIFIED]**.

## 1. Result

| Item | Value |
|---|---|
| Scanner | `pip-audit` 2.10.1 (separate virtualenv, not shipped) |
| Target | `pip freeze` of a **clean venv** created from the release's `requirements.txt` on Python 3.13.18 |
| Command | `pip-audit -r freeze.txt --no-deps --disable-pip -s pypi -f json` |
| Vulnerability service | `pypi` (the vulnerability data PyPI publishes per release) |
| Packages audited | **60** (all pinned exact versions, none skipped) |
| Known vulnerabilities | **0** |
| Run | [EXECUTED] 2026-11-22; JSON output delivered alongside the release reports (`pip_audit_final.json`) |

**Not verified:** the OSV service (`-s osv`) was also attempted and blocked by the
network allowlist (`api.osv.dev` → 403 from the egress proxy). Results reflect the
advisory data available through PyPI at scan time only; new advisories appear daily —
re-run `pip-audit -r requirements.txt` before deploying.

## 2. Dependency changes in this release

| Change | Reason |
|---|---|
| **Removed `anthropic>=0.18.0`** | No module imports it (verified by search of `src/`, `api/`, `dashboard/`, `app.py`, `tests/`). The AI Marketing Analyst is rule-based. Removing it also drops 6 transitive packages from the install (`docstring_parser`, `httpcore2`, `httpx2`, `jiter`, `sniffio`, `truststore`), reducing the audited surface from 67 to 60 packages. |
| No package added | — |
| Version ranges unchanged | `requirements.txt` declares minimums (`pytest` additionally `<9.2`, unchanged) |

Browser testing used Playwright and Google Chrome from the build environment; they are
tooling, not application dependencies, and are not in `requirements.txt`.

## 3. Verified installed set (clean venv, 60 packages)

```
altair==6.3.0 annotated-doc==0.0.7 annotated-types==0.8.0 anyio==4.16.0 attrs==26.1.0
certifi==2026.11.20 charset-normalizer==3.5.2 click==8.5.1 cloudpickle==3.1.3 et_xmlfile==2.0.0
fastapi==0.165.0 h11==0.16.0 httpcore==1.0.9 httptools==0.9.0 httpx==0.28.1 idna==3.20
iniconfig==2.3.1 itsdangerous==2.2.0 Jinja2==3.1.6 joblib==1.6.0 jsonschema==4.26.0
jsonschema-specifications==2025.9.1 MarkupSafe==3.0.4 narwhals==2.29.0 numpy==2.5.5
openpyxl==3.1.5 opentelemetry-api==1.45.2 packaging==26.3 pandas==3.1.1 pillow==13.0.0
plotly==7.1.0 pluggy==1.7.0 protobuf==7.36.2 pyarrow==26.0.1 pydantic==2.14.4
pydantic_core==2.50.5 pydeck==0.9.3 Pygments==2.21.0 pytest==9.1.1 python-dateutil==4.0.0
python-dotenv==1.2.4 python-multipart==0.0.34 PyYAML==6.0.4 referencing==0.37.0
requests==2.34.2 rpds-py==2026.9.1 scikit-learn==1.9.1 scipy==1.18.1 starlette==1.10.0
streamlit==1.67.0 threadpoolctl==3.8.0 typing-inspection==0.4.5 typing_extensions==4.16.0
tzdata==2026.6 urllib3==2.8.0 uvicorn==0.54.0 uvloop==0.23.0 watchdog==7.1.1
watchfiles==1.3.0 websockets==17.4
```

This exact set ran the full test suite (`FINAL_TESTING_REPORT.md`) and the application
(`FINAL_DEPLOYMENT_VERIFICATION.md`).

## 4. Recommendations for operators

- Pin this set (e.g. `pip freeze > constraints.txt`) for reproducible deployments;
  `requirements.txt` only declares minimums, so a later install may pull newer versions
  that were not tested.
- Re-run `pip-audit` (both PyPI and OSV services) in CI before each deployment.
- Rebuild Docker images regularly to pick up base-image security updates.
