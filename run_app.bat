@echo off
REM Financial Assistant AI - Quick Start Script
REM This script starts the Streamlit web interface

echo ========================================
echo  Financial Assistant AI
echo  Starting Streamlit Web Interface...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Starting Streamlit server...
echo.
echo The app will open in your default browser.
echo Access URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start Streamlit
streamlit run app.py

pause
