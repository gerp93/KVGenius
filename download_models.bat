@echo off
title KVGenius Model Downloader
cd /d "%~dp0"

echo ============================================================
echo KVGenius Model Downloader
echo ============================================================
echo.

:: Activate conda environment
call conda activate kvgen
if errorlevel 1 (
    echo ERROR: Failed to activate kvgen conda environment
    echo Make sure you have created the kvgen environment
    pause
    exit /b 1
)

echo Conda environment: kvgen activated
echo.

:menu
echo ============================================================
echo What would you like to download?
echo ============================================================
echo.
echo   [1] Download a specific model (interactive)
echo   [2] Download all chat models (skips existing)
echo   [3] Download all image models (skips existing)
echo   [4] List all available models
echo   [5] Force re-download all chat models
echo   [6] Force re-download all image models
echo   [Q] Quit
echo.
set /p choice="Enter choice: "

if /i "%choice%"=="1" (
    echo.
    python scripts/download_model.py
    echo.
    pause
    goto menu
)
if /i "%choice%"=="2" (
    echo.
    python scripts/download_chat_models.py
    echo.
    pause
    goto menu
)
if /i "%choice%"=="3" (
    echo.
    python scripts/download_image_models.py
    echo.
    pause
    goto menu
)
if /i "%choice%"=="4" (
    echo.
    python scripts/download_model.py --list
    echo.
    pause
    goto menu
)
if /i "%choice%"=="5" (
    echo.
    echo Force re-downloading all chat models...
    python scripts/download_chat_models.py --force
    echo.
    pause
    goto menu
)
if /i "%choice%"=="6" (
    echo.
    echo Force re-downloading all image models...
    python scripts/download_image_models.py --force
    echo.
    pause
    goto menu
)
if /i "%choice%"=="q" (
    echo Goodbye!
    exit /b 0
)

echo Invalid choice, try again.
goto menu
