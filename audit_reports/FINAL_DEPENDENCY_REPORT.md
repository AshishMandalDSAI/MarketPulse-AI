> **SUPERSEDED (2026-11-22).** Historical record of the 2026-11-21 audit of MarketPulse_AI_FINAL_RELEASE.zip, kept for traceability. The current, verified results are in the project root: `FINAL_RELEASE_STATUS.md`, `FINAL_TESTING_REPORT.md`, `FINAL_SECURITY_AUDIT_REPORT.md`, `FINAL_DEPLOYMENT_VERIFICATION.md`, `FINAL_CHANGELOG.md`, `FINAL_KNOWN_LIMITATIONS.md`.

# MarketPulse AI — Final Dependency Report

**Date:** 2026-11-21

## Python Version

- **Tested with:** Python 3.13.18

## requirements.txt — Declared Ranges vs Installed Versions

| Package | Declared Range | Installed Version | Status |
|---|---|---|---|
| pandas | `>=2.0.0` | 3.1.1 | OK |
| numpy | `>=1.24.0` | 2.5.5 | OK |
| scikit-learn | `>=1.3.0` | 1.9.1 | OK |
| scipy | `>=1.11.0` | 1.18.1 | OK |
| plotly | `>=5.18.0` | 7.1.0 | OK |
| streamlit | `>=1.30.0` | 1.67.0 | OK |
| openpyxl | `>=3.1.0` | 3.1.5 | OK |
| fastapi | `>=0.109.0` | 0.165.0 | OK |
| pydantic | `>=2.5.0` | 2.14.4 | OK |
| uvicorn[standard] | `>=0.27.0` | 0.54.0 | OK |
| pytest | `>=7.4.0,<9.2` | 9.1.1 | OK |
| httpx | `>=0.26.0` | 0.28.1 | OK |
| anthropic | `>=0.18.0` | 2.12.0 | OK (optional) |

### pytest Version Constraint

The upper bound `<9.2` on pytest is intentional. pytest 9.2.0 rejects `pytest.approx` on nested data structures used in `tests/test_phase8_part2_engine.py`. The constraint ensures the full test suite passes without modification. Tested and verified with pytest 9.1.1.

## Transitive Dependencies (notable)

| Package | Installed Version | Required By |
|---|---|---|
| python-multipart | 0.0.34 | fastapi (file uploads) |
| pydantic_core | 2.50.5 | pydantic |

## Corrections from Prior Report

The following packages were listed in the prior dependency report but are **not** declared in `requirements.txt` and are **not** installed in a clean venv:

- ~~xlsxwriter~~ — Not a dependency of this project
- ~~statsmodels~~ — Not a dependency of this project
- ~~python-dotenv~~ — Not in requirements.txt

Additionally, several installed versions were reported incorrectly in the prior report (e.g. numpy was listed as 2.3.1 but actually installs as 2.5.5; fastapi was listed as 0.115.12 but actually installs as 0.165.0). This corrected report reflects the actual `pip install -r requirements.txt` output in a clean Python 3.13.18 venv.

## Model Compatibility

| Component | Created With | Running On | Status |
|---|---|---|---|
| mmm_model.pkl (Ridge) | scikit-learn 1.8.0 | scikit-learn 1.9.1 | WARNING — loads with InconsistentVersionWarning |
| mmm_model.pkl (StandardScaler) | scikit-learn 1.8.0 | scikit-learn 1.9.1 | WARNING — loads with InconsistentVersionWarning |

The version mismatch produces warnings but does not affect functionality.

## Dependency Conflicts

None. All packages resolve without version conflicts via `pip install -r requirements.txt`.

## Verdict: **PASS**
