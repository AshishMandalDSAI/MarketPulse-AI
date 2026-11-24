> **SUPERSEDED (2026-11-22).** Historical record of the 2026-11-21 audit of MarketPulse_AI_FINAL_RELEASE.zip, kept for traceability. The current, verified results are in the project root: `FINAL_RELEASE_STATUS.md`, `FINAL_TESTING_REPORT.md`, `FINAL_SECURITY_AUDIT_REPORT.md`, `FINAL_DEPLOYMENT_VERIFICATION.md`, `FINAL_CHANGELOG.md`, `FINAL_KNOWN_LIMITATIONS.md`. The 14 warnings described here were resolved in the final release (model re-serialised; profiler fixed).

# MarketPulse AI — Final Known Issues

**Date:** 2026-11-21

## Warnings (14 total, all non-critical)

### W-001 to W-012: scikit-learn InconsistentVersionWarning (12 occurrences)
- **Severity:** LOW
- **Description:** Bundled demo model `models/mmm_model.pkl` was serialized with scikit-learn 1.8.0 but loads on 1.9.1
- **Impact:** Warning messages in logs; no functional impact
- **Mitigation:** Re-serialize the model with current scikit-learn version, or suppress warning in production
- **Action required:** None (demo model only)

### W-013 to W-014: numpy RuntimeWarning (2 occurrences)
- **Severity:** LOW
- **Description:** `invalid value encountered in reduce` during anomaly detection tests with infinite values
- **Impact:** Expected behavior — tests verify that the system handles `inf` values gracefully
- **Mitigation:** None needed; this is correct behavior being tested
- **Action required:** None

## Deprecation Fixes Applied

### D-001: pandas `dayofweek` deprecation (FIXED)
- **File:** `src/prediction/preprocessing.py:123`
- **Fix:** `ts.dt.dayofweek` → `ts.dt.day_of_week`

### D-002: pandas `ExcelFile.parse` deprecation (FIXED)
- **File:** `tests/test_phase8_part4_prescriptive.py:327`
- **Fix:** `x.parse(...)` → `pd.read_excel(x, ...)`

## Design Limitations (Not Bugs)

1. **Non-obvious column names require manual mapping** — The confidence-based column role system intentionally requires HIGH confidence for auto-mapping. This prevents false positives.

2. **In-memory processing** — The platform processes data in-memory via pandas. Very large datasets (>1M rows) may require additional memory.

3. **Single-user design** — No built-in authentication or multi-tenancy. Intended for single-analyst or team deployment behind an auth proxy.

## Critical Issues

**None.**

## Verdict: **PASS** (with 14 non-critical warnings)
