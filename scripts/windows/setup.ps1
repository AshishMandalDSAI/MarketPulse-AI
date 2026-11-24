<#
.SYNOPSIS
    Creates the MarketPulse AI virtual environment and installs the dependencies (Windows).
.DESCRIPTION
    Run from the project folder:
        powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
        powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -WithOptional
    -WithOptional also installs XGBoost, LightGBM and CatBoost (requirements-optional.txt).
    Works in Windows PowerShell 5.1 and PowerShell 7+. Also runs under PowerShell 7 on Linux/macOS.
#>
param(
    [switch]$WithOptional,
    [string]$Python = "",
    [string]$VenvDir = ".venv"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
. (Join-Path $PSScriptRoot "common.ps1")

$py = Find-Python313 -Preferred $Python
$pyExe = $py[0]
$pyArgs = @()
if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }
Write-Host "Using Python: $($py -join ' ')"
& $pyExe @pyArgs -c "import sys; print(sys.version)"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir ..."
    & $pyExe @pyArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed (exit $LASTEXITCODE)" }
}
$vpy = Get-VenvPython -VenvDir $VenvDir
& $vpy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
if ($WithOptional) {
    & $vpy -m pip install -r requirements-optional.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Optional packages failed to install. The app still works; those models show as 'unavailable'."
    }
}
& $vpy -c "import streamlit, pandas, sklearn, fastapi; print('core imports OK: streamlit', streamlit.__version__, '| pandas', pandas.__version__)"
if ($LASTEXITCODE -ne 0) { throw "core import check failed" }
Write-Host "Setup complete. Start the dashboard with: scripts\windows\start_dashboard.ps1"
