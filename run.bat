@echo off
setlocal
cd /d "%~dp0"

echo ======================================================
echo   AI Fashion Accessorizer - Startup Script
echo ======================================================

echo.
echo [1/4] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo [2/4] Setting up Virtual Environment (venv)...
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Try running: python -m pip install --user virtualenv
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo [3/4] Installing/Updating dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies. 
    echo Please check your internet connection and try again.
    echo Common fix: Close any programs using these files and restart this script.
    pause
    exit /b 1
)

echo.
echo [4/4] Starting the Application...
echo.
echo ======================================================
echo   The app will be available at: http://127.0.0.1:8000
echo   Press CTRL+C to stop the server.
echo ======================================================
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application crashed or failed to start.
    pause
)

deactivate
