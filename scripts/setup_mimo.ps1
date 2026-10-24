$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$corePython = Join-Path $researchRoot '.venv\Scripts\python.exe'
$mimoVenv = Join-Path $researchRoot '.venv-mimo'
$mimoPython = Join-Path $mimoVenv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $corePython)) {
    throw "Research-root base Python not found: $corePython. Run scripts/setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $mimoPython)) {
    & $corePython -m venv $mimoVenv
    if ($LASTEXITCODE -ne 0) { throw 'MIMO venv creation failed.' }
}

& $mimoPython -m pip install -r (Join-Path $PSScriptRoot '..\requirements-mimo.txt')
if ($LASTEXITCODE -ne 0) { throw 'NumPy/pandas installation failed.' }
& $mimoPython -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU-only PyTorch installation failed.' }

& $mimoPython -c "import json, numpy, pandas, torch; print(json.dumps({'python':__import__('sys').version,'torch':torch.__version__,'torch_cuda_available':torch.cuda.is_available(),'numpy':numpy.__version__,'pandas':pandas.__version__},indent=2)); assert not torch.cuda.is_available(), 'Expected a CPU-only torch build'"
if ($LASTEXITCODE -ne 0) { throw 'MIMO environment verification failed.' }
Write-Output "Optional MIMO research environment ready at $mimoVenv"
