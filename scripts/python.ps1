param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Research-root venv not found: $pythonExe. Create it with python -m venv E:\Research\.venv."
}
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$PSScriptRoot\..\src;$env:PYTHONPATH" } else { "$PSScriptRoot\..\src" }
& $pythonExe @PythonArgs
exit $LASTEXITCODE
