@echo off
REM KVGenius Launcher
REM AI Chat + Image Generation on RTX 5070 Ti

echo ============================================
echo        KVGenius - AI Studio
echo   Chat + Image Gen + LoRA Training
echo   RTX 5070 Ti with sm_120 support
echo ============================================
echo.

cd /d "%~dp0"

echo Starting KVGenius...
echo Open http://127.0.0.1:7860 in your browser
echo.

"%USERPROFILE%\Miniconda3\envs\kvgen\python.exe" -u web_app_multi.py

pause
