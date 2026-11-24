# MarketPulse AI Upgrade: Part Status

The roadmap is in `IMPLEMENTATION_PLAN.md` (approved four-part plan).

| Part | Scope | Status | Tests | Backup |
|---|---|---|---|---|
| 00 | Baseline audit and backup | **Complete** | Baseline 3,030 / 3,030 passed | `MarketPulse_AI_PART00_BASELINE_BACKUP.zip`, SHA-256 `659ab2a0b6584e40913b4a6965e888830130efa055c70f6a77fbd031ae696898` (unchanged) |
| 01 | Navigation, unified Statistics page with three tabs, Executive Overview chart corrections, automatic currency detection | **Complete; approved** | 3,150 collected / 3,150 passed / 0 failed / 0 skipped / 0 not run (`TEST_REPORT.md`) | `MarketPulse_AI_PART01_BACKUP.zip`; SHA-256 in `PART01_SHA256.txt` |
| 02 | Universal dataset engine; Spend, Exposure and General modes; four universal marketing models; optimizer default-limit fix | **Complete** (PARTs 03–05 authorised in advance on 2026-11-23) | See `TEST_REPORT.md` (PART 02 section) and `PART02_TEST_LOG.txt` | `MarketPulse_AI_PART02_BACKUP.zip`; SHA-256 in `PART02_SHA256.txt` |
| 03 | Full Statistics implementation: Z-test fix, 12 classifiers, CV and holdout evaluation, comparison charts | **Complete** | See `TEST_REPORT.md` (PART 03 section) and `PART03_TEST_LOG.txt` | `MarketPulse_AI_PART03_BACKUP.zip`; SHA-256 in `PART03_SHA256.txt` |
| 04 | Feasibility and constraint engine, Scenario Simulator completion, Prescriptive Analytics across the three modes | **Complete** | See `TEST_REPORT.md` (PART 04 section) and `PART04_TEST_LOG.txt` | `MarketPulse_AI_PART04_BACKUP.zip`; SHA-256 in `PART04_SHA256.txt` |
| 05 | Final integration, project audit, chart/UI audit, security & performance, Docker, final release | **Complete** | 3,449 collected / 3,446 passed / 0 failed / 3 skipped / 0 not run / 1 warning (`FINAL_TEST_REPORT.md`) | `MarketPulse_AI_FINAL_RELEASE.zip`; SHA-256 in `SHA256_CHECKSUM.txt` |
| 06 | Final completion: real Kaggle file, Z-test verification, Docker + optional ML, Windows, performance, deeper security, final regression | **Complete** (official Docker image **blocked**; Windows **not tested on Windows**) | 3,522 collected / 3,522 passed / 0 failed / 0 skipped / 0 warnings (`FINAL_TEST_REPORT.md`) | `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`; SHA-256 in `SHA256_CHECKSUM.txt` |

## PART 06: final status

Done in PART 06:

- The real Kaggle file was validated; one defect was fixed (content scores read as spend).
- The Z-test was verified against hand-computed values.
- The opt-in optional-ML Docker build was added and tested on the stand-in base.
- Windows scripts and a validator were added and run under PowerShell 7 on Linux.
- *Confirm & Analyze* is 12–19% faster with identical results.
- 7 security findings were fixed.
- The full suite runs with 0 skipped and 0 warnings.

Still outstanding, all external: the official Docker base image (blocked), Windows execution, and an external
penetration test. See `FINAL_PROJECT_STATUS.md` §3.

## PART 05: final status

The full list of limitations is in `FINAL_PROJECT_STATUS.md` §4. Items carried forward from
earlier parts and resolved in PART 05:

- PART 01 item 4 (Forecast chart labels): **fixed**.
- The PART 01–04 backup ZIPs have directory mode 0600. This is a packaging defect found in
  PART 05; the final release is fixed and the old backups are left untouched. Workaround:
  `chmod -R u+rwX,go+rX` after extracting.

Docker was verified only with a local substitute base image. The real Kaggle file was never
provided.

## PART 02: limitations carried forward (by design or scheduled)

1. **The real Kaggle file wasn't tested.** It was never provided. Its checks run on a replica with the same structure, and 3 tests are skipped until the file is placed at `tests/fixtures/part02/real/` (see `KAGGLE_DATASET_TEST_REPORT.md`).
2. **The shared driver model has no carry-over (adstock).** On the bundled demo it is graded *weak* (holdout R² ≈ 0), and the pages say so. In spend mode the existing adstock / saturation MMM is shown alongside it.
3. **Long-format channel data** (a channel-name column plus a value column) is recognised as a channel dimension but isn't pivoted into channels.
4. ~~Optimizer: still to come in PART 04~~: reserve budget, diagnostics, remedies A–D and solution statuses were delivered in PART 04.
5. **Scenario Simulator** was completed in PART 04 (budget scenarios, sensitivity). Ranges remain approximate (holdout error only); full prediction intervals that include parameter uncertainty are not implemented.
6. **Demo mapping edits** last for the session only, because the demo is bundled read-only data.
7. **Protected files changed, with documented reasons:** `dashboard/pages/0_upload_data.py`, `src/data_engine/column_classifier.py`, `src/marketing/field_discovery.py`. The other 82 protected files are byte-identical.
8. **Verification platforms:** Linux and Chrome only.

## PART 01: limitations carried forward (by design or scheduled)

1. ~~Pages 04–07 transition pages~~: replaced by the implemented models in PART 02.
2. ~~Optimizer 55% default bug~~: fixed in PART 02 (`PART02_OPTIMIZER_FIX_REPORT.md`). Full feasibility handling comes in PART 04.
3. ~~Z-test refusal before σ is entered~~: fixed in PART 03.
4. ~~The Forecast page is protected and unchanged~~: its chart got a title, axis titles and a currency-aware y-axis in PART 05 (a documented change to a protected file).
5. **The Statistics modules' notes** ("delivered in Part *n* of Phase 8") were reviewed in PART 03 and kept, because existing tests assert them. They refer to the earlier Phase 8 project and are accurate as history.
6. **Verification platforms:** Linux and Chrome only. Windows and other browsers were not run.
