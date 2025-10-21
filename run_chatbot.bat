@echo off
REM KVGenius Chatbot Launcher
REM Ensures the correct environment and DLL paths are set

echo ============================================
echo KVGenius AI Chatbot
echo Running on RTX 5070 Ti with sm_120 support
echo ============================================
echo.

CALL "%USERPROFILE%\Miniconda3\Scripts\activate.bat" kvgen
if errorlevel 1 (
    echo ERROR: Failed to activate kvgen environment
    pause
    exit /b 1
)

echo Environment: kvgen activated
echo Starting chatbot...
echo.

cd /d "%~dp0"
python cli_app.py

pause
