<#
.SYNOPSIS
    Starts the MarketPulse AI dashboard (Streamlit) at http://localhost:8501. Stop it with Ctrl+C.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\start_dashboard.ps1 -Port 8501
#>
param([int]$Port = 8501, [string]$VenvDir = ".venv")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
. (Join-Path $PSScriptRoot "common.ps1")
$vpy = Get-VenvPython -VenvDir $VenvDir
$env:PYTHONUTF8 = "1"
& $vpy -m streamlit run app.py --server.port $Port --server.headless true
