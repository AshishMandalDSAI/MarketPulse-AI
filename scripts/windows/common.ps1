# Shared helpers for the MarketPulse AI PowerShell scripts (dot-sourced; not run directly).

function Find-Python313 {
    param([string]$Preferred = "")
    # Returns the command as an array, e.g. @("py", "-3.13") or @("python").
    $candidates = @()
    if ($Preferred) { $candidates += , @($Preferred) }
    $candidates += , @("py", "-3.13")
    $candidates += , @("python3.13")
    $candidates += , @("python")
    $candidates += , @("python3")
    foreach ($c in $candidates) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        try {
            $args2 = @()
            if ($c.Length -gt 1) { $args2 = $c[1..($c.Length - 1)] }
            $v = & $c[0] @args2 -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and "$v".Trim() -eq "3.13") { return , $c }
        } catch { }
    }
    throw "Python 3.13 was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), or pass -Python <path-to-python.exe>."
}

function Get-VenvPython {
    param([string]$VenvDir = ".venv")
    $win = Join-Path $VenvDir "Scripts\python.exe"
    $nix = Join-Path (Join-Path $VenvDir "bin") "python"
    if (Test-Path $win) { return (Resolve-Path $win).Path }
    if (Test-Path $nix) { return (Resolve-Path $nix).Path }
    throw "No virtual environment at '$VenvDir'. Run scripts\windows\setup.ps1 first."
}

function Wait-Http {
    param([string]$Url, [int]$TimeoutSec = 90)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    return $false
}

function Get-CurlExe {
    # Windows 10/11 ship curl.exe; on Linux/macOS it is plain "curl". Never the PowerShell alias.
    foreach ($n in @("curl.exe", "curl")) {
        $c = Get-Command $n -CommandType Application -ErrorAction SilentlyContinue
        if ($c) { return $c[0].Source }
    }
    return $null
}
