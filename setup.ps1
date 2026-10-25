# Cortex-M Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  CORTEX-M Automated PowerShell Installer               " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

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
    Write-Error "Python 3.9+ was not found on your system! Please install Python from https://www.python.org/downloads/"
    exit 1
}

Write-Host "[*] Found Python: $python" -ForegroundColor Green

if (-not (Test-Path "$PSScriptRoot\.venv") -and -not (Test-Path "$PSScriptRoot\..\.venv-mimo")) {
    Write-Host "[*] Creating dedicated virtual environment in .venv..." -ForegroundColor Yellow
    & $python -m venv "$PSScriptRoot\.venv"
    $python = "$PSScriptRoot\.venv\Scripts\python.exe"
}

Write-Host "[*] Upgrading pip..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip

Write-Host "[*] Installing Cortex-M package in editable mode..." -ForegroundColor Yellow
& $python -m pip install -e "$PSScriptRoot"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Installation Completed Successfully!                  " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host "Run .\run.ps1 or 'cortex-studio' to start." -ForegroundColor White
