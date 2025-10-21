@echo off
REM Build PyTorch from source with sm_120 (Blackwell) support

REM 1. Activate your conda environment
setlocal
set "PYTORCH_SRC=C:\Users\kgerp\source\repos\pytorch_src"
echo ============================================
echo PyTorch SM_120 Source Build Helper
echo ============================================
echo.
echo [1/8] Activating conda environment 'kvgen'...
CALL "%USERPROFILE%\Miniconda3\Scripts\activate.bat" kvgen
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment. Make sure Miniconda is installed and the env 'kvgen' exists.
    goto :end
)

REM 2. Install build dependencies
echo [2/8] Installing build dependencies (conda and pip)...
CALL conda install -y cmake ninja git || (
    echo ERROR: conda install failed
    goto :end
)
CALL pip install --upgrade pip || goto :end
CALL pip install numpy typing_extensions || goto :end

REM 3. Enable long paths in Git globally (CRITICAL FIX)
echo [3/8] Configuring Git for long paths...
git config --global core.longpaths true
if errorlevel 1 (
    echo WARNING: Failed to set git longpaths config
)

REM 4. Clone PyTorch source (if not already present)
echo [4/8] Ensuring PyTorch source repository exists...
if not exist "%PYTORCH_SRC%" (
    echo Cloning PyTorch repository to %PYTORCH_SRC% - this may take a while...
    git clone --recursive https://github.com/pytorch/pytorch.git "%PYTORCH_SRC%"
    if errorlevel 1 goto :end
)

pushd "%PYTORCH_SRC%"
echo [5/8] Updating submodules...
CALL git submodule sync || goto :pop_and_end
CALL git submodule update --init --recursive || goto :pop_and_end

REM 6. Set CUDA arch for Blackwell (sm_120)
echo [6/8] Configuring build environment for sm_120 (Blackwell)...
REM Try to detect a CUDA installation - adjust if your CUDA path is different
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
) else (
    set "CUDA_HOME=%CUDA_PATH%"
)
set "TORCH_CUDA_ARCH_LIST=12.0"
set "USE_CUDA=1"
echo   TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo   CUDA_HOME=%CUDA_HOME%

REM 7. Build PyTorch
echo [7/8] Building PyTorch from source (this will take several hours)...
REM Use develop for faster iterative installs; switch to 'install' if you prefer a site install
CALL python setup.py develop --cmake || (
    echo ERROR: PyTorch build failed
    goto :pop_and_end
)

REM 8. Verify installation
echo [8/8] Verifying PyTorch installation...
set "VERIFY_PY=%TEMP%\verify_torch.py"
echo import torch>"%VERIFY_PY%"
echo print('PyTorch:', torch.__version__ )>>"%VERIFY_PY%"
echo print('CUDA available:', torch.cuda.is_available() )>>"%VERIFY_PY%"
echo if torch.cuda.is_available():>>"%VERIFY_PY%"
echo     try:>>"%VERIFY_PY%"
echo         print('Device name:', torch.cuda.get_device_name(0))>>"%VERIFY_PY%"
echo         print('CUDA arch list:', torch.cuda.get_arch_list())>>"%VERIFY_PY%"
echo     except Exception as e:>>"%VERIFY_PY%"
echo         print('Runtime check failed:', e)>>"%VERIFY_PY%"

CALL python "%VERIFY_PY%"
if exist "%VERIFY_PY%" del "%VERIFY_PY%"

popd
echo.
echo ============================================
echo Build script finished!
echo If the build succeeded you should see sm_120 in the CUDA arch list above.
echo ============================================
goto :end

:pop_and_end
popd
echo ERROR: Aborting due to previous error.

:end
endlocal
goto :eof