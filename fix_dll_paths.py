"""
DLL Path Fix for Custom-Built PyTorch with sm_120 Support
Import this module at the top of your scripts BEFORE importing torch.
"""
import os
import sys

def setup_pytorch_dll_paths():
    """Add required DLL directories for custom-built PyTorch."""
    dll_paths = [
        r"C:\Users\kgerp\source\repos\pytorch_src\torch\lib",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\extras\CUPTI\lib64",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
    ]
    
    for path in dll_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                # AttributeError if Python < 3.8
                # OSError if path already added
                pass
    
    # Fix for PyTorch nightly build on Windows - set cache dir before torch._dynamo is imported
    if 'TORCHINDUCTOR_CACHE_DIR' not in os.environ:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch_inductor')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

# Auto-run when imported
setup_pytorch_dll_paths()
