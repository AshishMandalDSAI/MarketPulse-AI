# PART 02 changed-file manifest

Compared with `MarketPulse_AI_PART01_BACKUP.zip` (SHA-256 `808b4e13…7c52`). Nothing was removed.
SHA-256 values are for the files in `MarketPulse_AI_PART02_BACKUP.zip`.

## Modified (14)

| File | Reason | SHA-256 |
|---|---|---|
| `CHANGELOG.md` | Documentation update for PART 02. | `bd7e30029a9bbe48…` |
| `DATASET_COMPATIBILITY_REPORT.md` | Documentation update for PART 02. | `3dd5c6cecb3377f3…` |
| `IMPLEMENTATION_PLAN.md` | Documentation update for PART 02. | `0553dbbfeb4a155b…` |
| `PART_STATUS.md` | Documentation update for PART 02. | `8e7944c24545da1c…` |
| `TEST_REPORT.md` | Documentation update for PART 02. | `b788c537c1e6cacd…` |
| `dashboard/components/prescriptive_ui.py` | 55% default fix: feasible defaults, refresh vs manual, reset button, feasibility summary, solver status and objective report; `_constraint_inputs` gained optional `key_fn` and `unit` arguments (backward compatible). | `b8494866eb6f2e0d…` |
| `dashboard/pages/0_upload_data.py` | PROTECTED. Step 2 is now the universal role mapping with target confirmation; Confirm & Analyze waits for an ambiguous target; the mapping is saved; mapping money columns feed the currency record; the superseded `_render_roles` widget function was removed. | `a84e32462632a9b6…` |
| `dashboard/pages/channel_performance.py` | Transition page replaced by Model 05. | `2dfab1af672bb18b…` |
| `dashboard/pages/marketing_effectiveness.py` | Transition page replaced by Model 04. | `0c465c3ed1ed6277…` |
| `dashboard/pages/resource_optimizer.py` | Transition page replaced by Model 06. | `86deceaf320760de…` |
| `dashboard/pages/scenario_simulator.py` | Transition page replaced by Model 07. | `ec583cac2993259b…` |
| `src/data_engine/column_classifier.py` | PROTECTED. Skips trade / promotion spend names when guessing channels (5 lines plus 1 import). | `3139ae930c86eecf…` |
| `src/marketing/field_discovery.py` | PROTECTED. Same rule in the Phase-6 name-based channel guess (signal 3 only; confirmed channels unaffected; 5 lines plus 1 import). | `d08fea721b693484…` |
| `tests/test_upgrade_part01.py` | Interim-page test now asserts the implemented page (see TEST_REPORT.md). | `d13773f584764e3a…` |

## Added (34)

| File | SHA-256 |
|---|---|
| `KAGGLE_DATASET_TEST_REPORT.md` | `f60f543f37df83ee…` |
| `PART02_MODEL_IMPLEMENTATION_REPORT.md` | `3d3c319d97244d77…` |
| `PART02_OPTIMIZER_FIX_REPORT.md` | `4f0a49f1db26276b…` |
| `PART02_TEST_LOG.txt` | `8f49b84b6b49c277…` |
| `dashboard/components/model_page.py` | `0153bc9e5aa84a2d…` |
| `dashboard/components/role_mapping_ui.py` | `6eda54854adb59cf…` |
| `docs/part02/compatibility_details.md` | `a4c28422931720d0…` |
| `docs/part02/kaggle_structure_replica_report.md` | `4355b79ce77e57a8…` |
| `src/data_engine/role_engine.py` | `3cc3e525e0fdbfc5…` |
| `src/data_engine/role_lexicon.py` | `61250d2ec7688fff…` |
| `src/marketing_intel/__init__.py` | `e3b0c44298fc1c14…` |
| `src/marketing_intel/capabilities.py` | `901782ad93ecf371…` |
| `src/marketing_intel/channels.py` | `1ba06b285ec7d599…` |
| `src/marketing_intel/context.py` | `67a1b42192b31cc8…` |
| `src/marketing_intel/effectiveness.py` | `f7566e8e36e6deed…` |
| `src/marketing_intel/optimizer.py` | `f42c9caf864b2fee…` |
| `src/marketing_intel/response.py` | `1657e718f9fe7fcb…` |
| `src/marketing_intel/scenarios.py` | `02ba2efbec6a84cf…` |
| `src/marketing_intel/stats.py` | `307522360785ce16…` |
| `src/prescriptive/limits.py` | `460f0e7222f71e6f…` |
| `tests/fixtures/part02/brand_geo_spend.csv` | `a598137851cd4d3d…` |
| `tests/fixtures/part02/exposure_daily.csv` | `dfa7bbe6b6e0dd47…` |
| `tests/fixtures/part02/general_sales.csv` | `b569ed3d6c6a4e04…` |
| `tests/fixtures/part02/kaggle_structure_replica.csv` | `ac48cb270ba273ff…` |
| `tests/fixtures/part02/make_datasets.py` | `c0faf15b8ef13e6b…` |
| `tests/fixtures/part02/missing_marketing.csv` | `df5aa309d587a591…` |
| `tests/fixtures/part02/mixed_types.csv` | `98f420e8c4308b80…` |
| `tests/fixtures/part02/multi_revenue.csv` | `fe123478e45bcbd6…` |
| `tests/fixtures/part02/one_channel.csv` | `bf867e189d4648da…` |
| `tests/fixtures/part02/renamed_columns.csv` | `91f075fe8e0da55a…` |
| `tests/fixtures/part02/spend_weekly.csv` | `ed683ca2fd66e83a…` |
| `tests/fixtures/part02/two_channel.csv` | `22e500ae8ee951fb…` |
| `tests/test_upgrade_part02.py` | `7a7cfc18f96de2f4…` |
| `tools/part02_compat_report.py` | `55f74e8b814a8348…` |

## Protected files (PART 00 list of 85)

- Changed: `dashboard/pages/0_upload_data.py`, `src/data_engine/column_classifier.py`, `src/marketing/field_discovery.py` (reasons above).
- The other 82 are byte-identical to the baseline.
- Unchanged: the Forecast module, the models, security files, `requirements.txt`, the Dockerfiles and `api/main.py`.
- `docker-entrypoint.sh` keeps its executable bit.
