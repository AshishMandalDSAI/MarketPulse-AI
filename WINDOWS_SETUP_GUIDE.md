# Windows setup and validation guide

> **Status: NOT YET VERIFIED ON WINDOWS.** No Windows machine was available while building this
> release. Everything below was prepared for Windows, and the scripts were executed end to end with
> PowerShell 7.5.4 **on Linux** (§6). That proves the script logic, not Windows itself. Please run
> §4 on your Windows machine and keep the report it writes.

## 1. Requirements

| Item | Version / note |
|---|---|
| Windows | 10 (1803 or later) or 11, 64-bit. `curl.exe` is built in from 1803 onwards. |
| Python | **3.13**, 64-bit, from python.org. Tick *"Add python.exe to PATH"*. The `py` launcher is used when present. |
| PowerShell | Windows PowerShell 5.1 (built in) or PowerShell 7+. The scripts are ASCII-only and work in both. |
| Disk | About 1.5 GB for the virtual environment with the optional models, and about 0.6 GB without them. |
| Docker (optional) | Docker Desktop with the WSL 2 backend. See `DOCKER_SETUP_GUIDE.md`. |

## 2. Extract

1. Right-click `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`, choose **Properties**, tick **Unblock**
   if it is shown, and click OK. This prevents "script is not digitally signed" warnings.
2. Choose **Extract All…**, preferably to a short path such as `C:\MarketPulse\`.
   The longest path in the release is about 160 characters with a typical `Downloads` prefix,
   well under the 260-character Windows limit.
3. Optionally verify the ZIP:
   ```powershell
   Get-FileHash .\MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip -Algorithm SHA256
   ```
   Compare the result with the first line of `SHA256_CHECKSUM.txt`. The comparison ignores case.

## 3. Install and run (PowerShell)

```powershell
cd C:\MarketPulse\MarketPulse_AI_FINAL_RELEASE
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -WithOptional
powershell -ExecutionPolicy Bypass -File scripts\windows\start_dashboard.ps1
```

The dashboard opens at **http://localhost:8501**; stop it with **Ctrl+C**.

To start the API, open a second PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start_api.ps1
```

The API runs at **http://127.0.0.1:8000/docs**.

If you prefer to work without the scripts:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt   # optional models
streamlit run app.py
```

| Linux / macOS command | Windows equivalent |
|---|---|
| `source venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| `cp .env.example .env` | `Copy-Item .env.example .env` |
| `export VAR=value` | `$env:VAR = "value"` |
| `sha256sum file` | `Get-FileHash file -Algorithm SHA256` |
| `rm -rf folder` | `Remove-Item folder -Recurse -Force` |
| `curl -F file=@x.csv …` | `curl.exe -F "file=@x.csv" …` (use `curl.exe`, not the `curl` alias) |
| `kill <pid>` / Ctrl+C | `Stop-Process -Id <pid>` / Ctrl+C |

## 4. One-command validation (please run this on Windows)

```powershell
cd C:\MarketPulse\MarketPulse_AI_FINAL_RELEASE
powershell -ExecutionPolicy Bypass -File scripts\windows\validate_windows.ps1
```

The full run takes about 20–30 minutes, including about 14 minutes of tests. Options:

- `-Quick` runs 4 test files instead of the whole suite, in about 3 minutes.
- `-NoOptional` skips XGBoost, LightGBM and CatBoost.
- `-SkipSetup` reuses an existing `.venv`.

The script writes **`windows_validation_report.md`** (and `windows_test_log.txt`) in the project
folder, and exits with code 0 only when every step passed.

| Step | What is checked | Expected output |
|---|---|---|
| 1 setup | venv creation, `pip install` of the core and optional requirements | `[PASS] 1 setup` |
| 1b python | Python 3.13 in the venv | `[PASS] 1b python - 3.13.x Windows-10/11…` |
| 2 smoke: environment | OS, encoding (cp1252 is normal on Windows), project path | PASS; `"utf8_mode": 0` is expected |
| 2 smoke: paths_and_write_permissions | Windows paths under `data\uploads`; a UTF-8 file with a space and `₹` in its name; rejects `..\..\Windows`, `C:\Windows` and `a/b` as dataset IDs | PASS, `"separator": "\\"` |
| 2 smoke: csv_and_excel_upload_parsing | The same CSV read as `.csv` and as `.xlsx` (path with a space) gives identical data | PASS, equal shapes |
| 2 smoke: role_engine_and_marketing_models | The real Kaggle file (included) is detected as Exposure-Based with no spend channels, and the model fits | PASS, `"mode": "exposure"` |
| 2 smoke: predictive_models_and_optional_libraries | All 12 classifiers run; each optional library is either used or reported *unavailable* | PASS; with `-NoOptional`, the three show `false` and are skipped, not failed |
| 2 smoke: report_generation_page | Reports page renders with its download buttons | PASS |
| 2 smoke: all_pages_render_demo | All 16 pages render | PASS, `"pages": 16` |
| 3 pytest (full) | The whole suite **with the Windows default encoding** (UTF-8 mode deliberately not forced) | `3522 passed` with 0 failed and 0 skipped, the Linux result in `FINAL_TEST_REPORT.md`. On Windows, expect these skips: the bash-based entrypoint tests (`TestSentinelLogic`, 8 tests) and the root-only privilege-drop tests (`TestEntrypointPrivilegeDrop`). Both exercise the Linux container entrypoint. Any other skip or failure is a real Windows finding: please send the log. |
| 4 data folder writable | Writes and deletes a file in `data\uploads` | PASS |
| 5a–5d API | Start, `/health`; upload without a key returns **401**; CSV and XLSX uploads with a key return `dataset_id`; `..\..\Windows` as a dataset ID returns **400** | PASS |
| 6 dashboard | Start, health check, home page returns HTTP 200 | PASS |
| 7a/7b shutdown | Both processes stopped and ports released | PASS |

**Please send back** `windows_validation_report.md` and `windows_test_log.txt` if any step fails.

## 5. Windows-specific notes

- **Antivirus / SmartScreen:** first start-up can be slow while Defender scans the new `.venv`.
- **Firewall prompt:** Windows may ask whether Python can accept connections. Choosing *Private*
  networks is enough for local use.
- **Ports in use:** pass `-Port` to the start scripts, or `-ApiPort` / `-DashPort` to the validator.
- **Long paths:** extract to a short folder if you see `FileNotFoundError` for very long paths.
- **Encoding:** the start scripts set `PYTHONUTF8=1`, and the app writes every non-ASCII file as
  UTF-8 explicitly. The test run deliberately does *not* force UTF-8, so the validation covers the
  Windows default (cp1252).
- **LightGBM** on Windows bundles its OpenMP runtime; no extra install is needed.
- **Line endings:** `.gitattributes` keeps `*.sh` files LF if you put the project in Git. A CRLF
  `docker-entrypoint.sh` breaks the Docker containers.

## 6. What was verified, and where

| Check | Where it ran | Result |
|---|---|---|
| Script syntax (all 6 `.ps1`) | PowerShell 7.5.4 parser (Linux) | 0 parse errors |
| `validate_windows.ps1` end to end, including `setup.ps1` (venv + core + optional install), smoke, tests, API, dashboard and shutdown | PowerShell 7.5.4 on **Linux** | All steps PASS. Two portability bugs found this way were fixed: hard-coded `\` paths and the Windows-only `-WindowStyle` |
| File names: reserved names, invalid characters, case collisions, path length | Static scan of all release files | No issues; the longest path is 61 characters relative |
| Code that assumes Linux (`fcntl`, `resource`, `os.getuid`, `/tmp`, shell-only calls) in app code | Static scan | None found |
| Text-file encoding (UTF-8 versus the Windows default) | Static scan, plus the full suite under a strict non-UTF-8 locale on Linux | See `FINAL_TEST_REPORT.md` §3 |
| **Python 3.13 on Windows, PowerShell 5.1, Windows paths, Excel, Defender or firewall behaviour** | **Not run: no Windows machine** | **NOT VERIFIED.** Run §4 to verify. |
