@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   CORTEX-M Automated Installer for Windows
echo ========================================================
echo.

:: Detect Python
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
    echo [ERROR] Python 3.9+ was not found on your system!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [*] Using Python: %PYTHON_CMD%

:: Create virtual environment if neither exists
if not exist ".venv" (
    if not exist "..\.venv-mimo" (
        echo [*] Creating virtual environment in .venv...
        %PYTHON_CMD% -m venv .venv
        set PYTHON_CMD=.venv\Scripts\python.exe
    )
)

:: Install/Upgrade pip and requirements
echo [*] Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip

echo [*] Installing Cortex-M and dependencies...
%PYTHON_CMD% -m pip install -e .

if %errorlevel% neq 0 (
    echo [WARNING] Direct package install had warnings, installing requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

echo.
echo ========================================================
echo   CORTEX-M Installation Completed Successfully!
echo ========================================================
echo.
echo You can now start Cortex Studio anytime by running:
echo   run.bat
echo or
echo   cortex-studio
echo.
pause
