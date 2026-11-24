<#
.SYNOPSIS
    ONE-COMMAND Windows validation of MarketPulse AI. Writes windows_validation_report.md.
.DESCRIPTION
    From the extracted project folder, in PowerShell:
        powershell -ExecutionPolicy Bypass -File scripts\windows\validate_windows.ps1
    Options:
        -Quick          run only the fast test files instead of the full suite (~15 min)
        -NoOptional     do not install XGBoost / LightGBM / CatBoost
        -SkipSetup      reuse an existing .venv
    Steps (each recorded PASS / FAIL / SKIP, nothing assumed):
        1 setup (venv + requirements)          5 API: start, health, CSV + XLSX upload, auth, invalid id, stop
        2 platform smoke (tools/platform_smoke.py)  6 dashboard: start, health, home page, stop
        3 test suite (pytest)                  7 ports released after shutdown
        4 file permissions in data\
    Exit code 0 only when every step passed.
#>
param([switch]$Quick, [switch]$NoOptional, [switch]$SkipSetup, [string]$Python = "",
      [int]$ApiPort = 8765, [int]$DashPort = 8766)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
. (Join-Path $PSScriptRoot "common.ps1")
$Report = Join-Path $Root "windows_validation_report.md"
$Steps = New-Object System.Collections.ArrayList
function Add-Step([string]$Name, [string]$Result, [string]$Detail) {
    [void]$Steps.Add([pscustomobject]@{ Step = $Name; Result = $Result; Detail = ($Detail -replace "\r?\n", " ") })
    $color = @{ PASS = "Green"; FAIL = "Red"; SKIP = "Yellow" }[$Result]
    Write-Host ("[{0}] {1} - {2}" -f $Result, $Name, $Detail) -ForegroundColor $color
}
$started = Get-Date

# ---------------------------------------------------------------- 1. setup
try {
    if ($SkipSetup) { Add-Step "1 setup" "SKIP" "-SkipSetup given" }
    else {
        $a = @{ Python = $Python }
        if (-not $NoOptional) { $a["WithOptional"] = $true }
        & (Join-Path $PSScriptRoot "setup.ps1") @a
        Add-Step "1 setup" "PASS" "venv created and requirements installed"
    }
    $vpy = Get-VenvPython
} catch { Add-Step "1 setup" "FAIL" $_.Exception.Message; $vpy = $null }

if ($vpy) { try {
    $pyver = & $vpy -c "import sys,platform; print(sys.version.split()[0], platform.platform())"
    Add-Step "1b python" ($(if ("$pyver" -like "3.13*") { "PASS" } else { "FAIL" })) "$pyver"

    # ------------------------------------------------------------ 2. platform smoke
    $smokeJson = Join-Path $Root "platform_smoke.json"
    & $vpy (Join-Path "tools" "platform_smoke.py") $smokeJson
    $smokeOk = ($LASTEXITCODE -eq 0)
    if (Test-Path $smokeJson) {
        $sm = Get-Content $smokeJson -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($r in $sm.results) { Add-Step ("2 smoke: " + $r.check) $r.result (($r.detail | ConvertTo-Json -Compress -Depth 4)) }
    } else { Add-Step "2 platform smoke" "FAIL" "tools/platform_smoke.py produced no result (exit $LASTEXITCODE)" }

    # ------------------------------------------------------------ 3. tests
    & $vpy -m pip show pytest *> $null
    if ($LASTEXITCODE -ne 0) { & $vpy -m pip install "pytest>=8.0,<9.2" | Out-Null }
    $log = Join-Path $Root "windows_test_log.txt"
    $target = @("tests")
    if ($Quick) { $target = @("test_upgrade_part06.py", "test_api.py", "test_upgrade_part04.py", "test_ultimate_security.py") | ForEach-Object { Join-Path "tests" $_ } }
    & $vpy -m pytest @target -q -rs -p no:cacheprovider 2>&1 | Tee-Object -FilePath $log | Out-Null
    $code = $LASTEXITCODE
    $summary = (Get-Content $log -Encoding UTF8 | Select-String -Pattern "passed|failed|error" | Select-Object -Last 1).Line
    Add-Step "3 pytest $(if ($Quick) {'(quick)'} else {'(full)'})" ($(if ($code -eq 0) { "PASS" } else { "FAIL" })) "exit $code; $summary; log: windows_test_log.txt"

    # ------------------------------------------------------------ 4. permissions
    try {
        $probe = Join-Path (Join-Path (Join-Path $Root "data") "uploads") "perm_probe_$([guid]::NewGuid().ToString('N')).txt"
        Set-Content -Path $probe -Value "ok" -Encoding UTF8
        Remove-Item $probe
        Add-Step "4 data folder writable" "PASS" (Split-Path -Parent $probe)
    } catch { Add-Step "4 data folder writable" "FAIL" $_.Exception.Message }

    # ------------------------------------------------------------ 5. API
    $curl = Get-CurlExe
    $env:PYTHONUTF8 = "1"
    $env:MARKETPULSE_API_KEY = "win-validate-" + [guid]::NewGuid().ToString("N")
    $apiOut = Join-Path $Root "mp_api_out.txt"; $apiErr = Join-Path $Root "mp_api_err.txt"
    $sp = @{ FilePath = $vpy; PassThru = $true; RedirectStandardOutput = $apiOut; RedirectStandardError = $apiErr
             ArgumentList = @("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort", "--workers", "1") }
    if (-not ($IsLinux -or $IsMacOS)) { $sp["WindowStyle"] = "Hidden" }      # -WindowStyle exists only on Windows
    $api = Start-Process @sp
    try {
        if (Wait-Http "http://127.0.0.1:$ApiPort/health" 90) { Add-Step "5a API start + /health" "PASS" "port $ApiPort, pid $($api.Id)" }
        else { Add-Step "5a API start + /health" "FAIL" "no 200 within 90 s; see $apiErr" }
        if ($curl) {
            $csv = (Resolve-Path (Join-Path (Join-Path (Join-Path "tests" "fixtures") "part02") "spend_weekly.csv")).Path
            $xlsx = Join-Path (Join-Path (Join-Path $Root "data") "uploads") "win_validate.xlsx"
            & $vpy -c "import pandas as pd,sys; pd.read_csv(sys.argv[1]).to_excel(sys.argv[2], index=False)" $csv $xlsx
            $nullDev = "NUL"; if ($IsLinux -or $IsMacOS) { $nullDev = "/dev/null" }
            $noKey = & $curl -s -o $nullDev -w "%{http_code}" -F "file=@$csv" "http://127.0.0.1:$ApiPort/upload"
            Add-Step "5b upload without API key is rejected" ($(if ("$noKey" -eq "401") { "PASS" } else { "FAIL" })) "HTTP $noKey (expected 401)"
            foreach ($f in @($csv, $xlsx)) {
                $body = & $curl -s -H "X-API-Key: $($env:MARKETPULSE_API_KEY)" -F "file=@$f" "http://127.0.0.1:$ApiPort/upload"
                $ok = "$body" -match '"dataset_id"'
                Add-Step ("5c upload " + [IO.Path]::GetExtension($f)) ($(if ($ok) { "PASS" } else { "FAIL" })) ("$body".Substring(0, [Math]::Min(160, "$body".Length)))
            }
            Remove-Item $xlsx -ErrorAction SilentlyContinue
            $bad = & $curl -s -w " HTTP %{http_code}" -H "X-API-Key: $($env:MARKETPULSE_API_KEY)" "http://127.0.0.1:$ApiPort/kpis?dataset_id=..%5C..%5CWindows"
            Add-Step "5d invalid dataset id rejected" ($(if ("$bad" -match "HTTP 4\d\d") { "PASS" } else { "FAIL" })) "$bad"
        } else { Add-Step "5b-5d API uploads" "SKIP" "curl.exe not found (Windows 10 1803+ includes it)" }
    } finally {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Remove-Item Env:\MARKETPULSE_API_KEY -ErrorAction SilentlyContinue
        Remove-Item $apiOut, $apiErr -ErrorAction SilentlyContinue
    }
    $still = $true; try { Invoke-WebRequest "http://127.0.0.1:$ApiPort/health" -UseBasicParsing -TimeoutSec 3 | Out-Null } catch { $still = $false }
    Add-Step "7a API stopped, port released" ($(if (-not $still) { "PASS" } else { "FAIL" })) "port $ApiPort"

    # ------------------------------------------------------------ 6. dashboard
    $dOut = Join-Path $Root "mp_dash_out.txt"; $dErr = Join-Path $Root "mp_dash_err.txt"
    $sp = @{ FilePath = $vpy; PassThru = $true; RedirectStandardOutput = $dOut; RedirectStandardError = $dErr
             ArgumentList = @("-m", "streamlit", "run", "app.py", "--server.port", "$DashPort", "--server.headless", "true") }
    if (-not ($IsLinux -or $IsMacOS)) { $sp["WindowStyle"] = "Hidden" }
    $dash = Start-Process @sp
    try {
        if (Wait-Http "http://localhost:$DashPort/_stcore/health" 120) {
            $homePage = Invoke-WebRequest "http://localhost:$DashPort/" -UseBasicParsing -TimeoutSec 20
            Add-Step "6 dashboard start + health + home page" ($(if ($homePage.StatusCode -eq 200) { "PASS" } else { "FAIL" })) "port $DashPort, HTTP $($homePage.StatusCode)"
        } else { Add-Step "6 dashboard start + health + home page" "FAIL" "no 200 within 120 s; see mp_dash_err.txt" }
    } finally { Stop-Process -Id $dash.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2 }
    $still = $true; try { Invoke-WebRequest "http://localhost:$DashPort/_stcore/health" -UseBasicParsing -TimeoutSec 3 | Out-Null } catch { $still = $false }
    Add-Step "7b dashboard stopped, port released" ($(if (-not $still) { "PASS" } else { "FAIL" })) "port $DashPort"
    Remove-Item $dOut, $dErr -ErrorAction SilentlyContinue
} catch { Add-Step "unexpected error" "FAIL" ($_.Exception.Message + " at line " + $_.InvocationInfo.ScriptLineNumber) } }

# ---------------------------------------------------------------- report
$fail = @($Steps | Where-Object { $_.Result -eq "FAIL" }).Count
$lines = @("# MarketPulse AI - Windows validation report", "",
    "- Machine: $([Environment]::MachineName) | OS: $([Environment]::OSVersion.VersionString) | PowerShell $($PSVersionTable.PSVersion)",
    "- Started: $started | Finished: $(Get-Date) | Project: $Root", "",
    "| Step | Result | Detail |", "|---|---|---|")
foreach ($s in $Steps) { $lines += "| $($s.Step) | $($s.Result) | $(($s.Detail -replace '\|', '/').Substring(0, [Math]::Min(300, $s.Detail.Length))) |" }
$lines += ""; $lines += "**Overall: $(if ($fail -eq 0) {'ALL STEPS PASSED'} else {"$fail STEP(S) FAILED"})**"
$lines | Set-Content -Path $Report -Encoding UTF8
Write-Host "`nReport written to $Report"
if ($fail -eq 0) { exit 0 } else { exit 1 }
