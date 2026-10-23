$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$venvPath = Join-Path $researchRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPath)) {
    $systemPython = (Get-Command python -ErrorAction Stop).Source
    & $systemPython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'Python venv creation failed.' }
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Expected Python executable was not created: $pythonExe"
}
& $pythonExe --version
Write-Output "Cortex-M environment ready at $venvPath (standard library only)."
