<#
.SYNOPSIS
    Runs the MarketPulse AI test suite and saves the full log.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\run_tests.ps1
    powershell -ExecutionPolicy Bypass -File scripts\windows\run_tests.ps1 -Pattern "test_upgrade_part06.py"
#>
param([string]$Pattern = "", [string]$VenvDir = ".venv", [string]$Log = "windows_test_log.txt")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
. (Join-Path $PSScriptRoot "common.ps1")
$vpy = Get-VenvPython -VenvDir $VenvDir
& $vpy -m pip show pytest *> $null
if ($LASTEXITCODE -ne 0) { & $vpy -m pip install "pytest>=8.0,<9.2" }
$target = "tests"
if ($Pattern) { $target = Join-Path "tests" $Pattern }
# Deliberately NOT setting PYTHONUTF8: the tests must also pass with the Windows default encoding.
& $vpy -m pytest $target -v -rs -p no:cacheprovider 2>&1 | Tee-Object -FilePath $Log
$code = $LASTEXITCODE
Write-Host "pytest exit code: $code (0 = all passed). Full log: $Log"
exit $code
