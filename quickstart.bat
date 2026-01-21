@echo off
REM Quick Start Script for Windows
REM Environmental Health Advisory System

echo ============================================
echo Health Advisory System - Quick Start
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [3/4] Configuring API key...
if not exist .streamlit (
    mkdir .streamlit
)
if not exist .streamlit\secrets.toml (
    echo INFO: Create .streamlit/secrets.toml with:
    echo OWM_KEY = "your-openweathermap-api-key"
    echo https://openweathermap.org/api
    echo.
)

echo [4/4] Training ML model...
python train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next: streamlit run app.py
echo Browser: http://localhost:8501
echo.
pause
