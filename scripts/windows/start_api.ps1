<#
.SYNOPSIS
    Starts the MarketPulse AI API (FastAPI) at http://127.0.0.1:8000 (docs at /docs). Stop it with Ctrl+C.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\start_api.ps1 -Port 8000
#>
param([int]$Port = 8000, [string]$VenvDir = ".venv")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
. (Join-Path $PSScriptRoot "common.ps1")
$vpy = Get-VenvPython -VenvDir $VenvDir
$env:PYTHONUTF8 = "1"
& $vpy -m uvicorn api.main:app --host 127.0.0.1 --port $Port --workers 1
