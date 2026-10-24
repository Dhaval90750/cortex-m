@echo off
setlocal
set "PYTHONPATH=E:\Research\Cortex-M\src;%PYTHONPATH%"
"e:\Research\.venv-mimo\Scripts\python.exe" "E:\Research\Cortex-M\scripts\cortex_cli.py" %*
endlocal
