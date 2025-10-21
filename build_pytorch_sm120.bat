@echo off
REM PyTorch Source Build Script for SM_120 (RTX 5070 Ti)
REM This will take 4-6 hours to complete

echo ============================================
echo PyTorch SM_120 Source Build
echo ============================================
echo.

REM Step 1: Initialize Visual Studio 2022 environment
echo [1/6] Initializing Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: Failed to initialize Visual Studio environment
    exit /b 1
)
echo Visual Studio environment ready
echo.

REM Step 2: Activate conda environment
echo [2/6] Activating kvgen conda environment...
call conda activate kvgen
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    exit /b 1
)
echo Conda environment activated
echo.

REM Step 3: Install build dependencies
echo [3/6] Installing build dependencies...
pip install cmake ninja pyyaml setuptools typing_extensions
conda install -y mkl mkl-include
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed
echo.

REM Step 4: Clone PyTorch (if not already cloned)
echo [4/6] Setting up PyTorch source...
if not exist pytorch (
    echo Cloning PyTorch repository...
    git clone --recursive https://github.com/pytorch/pytorch.git
    if errorlevel 1 (
        echo ERROR: Failed to clone PyTorch
        exit /b 1
    )
) else (
    echo PyTorch directory already exists, updating...
    cd pytorch
    git pull
    git submodule sync
    git submodule update --init --recursive
    cd ..
)
echo.

REM Step 5: Configure build environment for SM_120
echo [5/6] Configuring build for SM_120...
cd pytorch

REM Set CUDA architecture to sm_120 (Blackwell/RTX 5070 Ti)
set TORCH_CUDA_ARCH_LIST=12.0
set USE_CUDA=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDNN_LIB_DIR=%CUDA_HOME%\lib\x64
set CUDNN_INCLUDE_DIR=%CUDA_HOME%\include
set CMAKE_BUILD_TYPE=Release

echo Build configuration:
echo   TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo   USE_CUDA=%USE_CUDA%
echo   CUDA_HOME=%CUDA_HOME%
echo   CMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
echo.

REM Step 6: Build PyTorch (this takes 4-6 hours)
echo [6/6] Building PyTorch from source...
echo WARNING: This will take 4-6 hours. Do not close this window.
echo Build started at %time%
echo.

python setup.py develop
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo ============================================
echo Build completed successfully at %time%
echo ============================================
echo.
echo Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA arch list:', torch.cuda.get_arch_list())"

echo.
echo Build script complete!
pause
