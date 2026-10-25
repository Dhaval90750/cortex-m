@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Launching CORTEX-M Studio...
echo ========================================================
echo.

set PYTHON_CMD=
for %%P in ("..\.venv-mimo\Scripts\python.exe" ".venv\Scripts\python.exe" "python" "py") do (
    if not defined PYTHON_CMD (
        %%~P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set PYTHON_CMD=%%~P
        )
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python not found. Please run setup.bat first!
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo [*] Starting Cortex Studio on http://127.0.0.1:8888 ...
%PYTHON_CMD% -m cortex_m.server.app

pause
