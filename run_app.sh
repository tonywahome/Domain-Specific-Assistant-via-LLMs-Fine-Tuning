#!/bin/bash
# Financial Assistant AI - Quick Start Script
# This script starts the Streamlit web interface

echo "========================================"
echo " Financial Assistant AI"
echo " Starting Streamlit Web Interface..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/3] Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
else
    echo "Dependencies OK"
fi

echo ""
echo "[2/3] Starting Streamlit server..."
echo ""
echo "The app will open in your default browser."
echo "Access URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start Streamlit
streamlit run app.py
