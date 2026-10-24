param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $researchRoot '.venv-mimo\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Optional MIMO environment not found: $pythonExe. Run scripts/setup_mimo.ps1."
}
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$PSScriptRoot\..\src;$env:PYTHONPATH" } else { "$PSScriptRoot\..\src" }
& $pythonExe @PythonArgs
exit $LASTEXITCODE
