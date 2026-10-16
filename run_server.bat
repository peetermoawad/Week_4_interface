@echo off
chcp 65001 > nul
title AquaSeg - Water Body Segmentation Server

echo ======================================================================
echo    🌊 AquaSeg AI - Water Body Segmentation Server 🛰️
echo ======================================================================
echo.

:: Ensure current working directory is the script's directory
cd /d "%~dp0"

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python and check 'Add Python to PATH'.
    pause
    exit /b 1
)

:: Check if weights exist
if not exist "weights\best_model.pth" (
    echo [WARNING] weights\best_model.pth not found!
    echo The API will start, but please ensure model weights are in place.
) else (
    echo [OK] Model weights found (best_model.pth)
)

echo.
echo [1/2] Starting AquaSeg Flask Server on http://localhost:5000 ...
echo [2/2] Opening browser automatically in 2 seconds...
echo.
echo ----------------------------------------------------------------------
echo  Press Ctrl+C in this window to stop the server anytime.
echo ----------------------------------------------------------------------
echo.

:: Open default browser in parallel after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Start Flask app
python app.py

pause
