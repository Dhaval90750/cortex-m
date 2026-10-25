# Cortex-M Launcher Script for Windows PowerShell
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting CORTEX-M Studio...                           " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

function Find-Python {
    $candidates = @(
        "$PSScriptRoot\..\.venv-mimo\Scripts\python.exe",
        "$PSScriptRoot\.venv\Scripts\python.exe",
        "python",
        "py"
    )
    foreach ($cand in $candidates) {
        try {
            $ver = & $cand --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                return $cand
            }
        } catch {}
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Error "Python not found. Please run .\setup.ps1 first!"
    exit 1
}

$env:PYTHONPATH = "$PSScriptRoot\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH

Write-Host "[*] Launching Cortex Studio server at http://127.0.0.1:8888 ..." -ForegroundColor Green
& $python -m cortex_m.server.app
