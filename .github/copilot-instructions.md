# AI Chatbot with Hugging Face - Project Instructions

## Project Overview
This is a Python-based AI chatbot application using Hugging Face transformers models for conversational AI, running on RTX 5070 Ti with custom-built PyTorch for sm_120 (Blackwell) support.

## CRITICAL: Environment Setup
**⚠️ ALWAYS USE THE `kvgen` CONDA ENVIRONMENT FOR ALL OPERATIONS**

### Running Commands
- **NEVER run commands in `base` environment**
- **ALWAYS use:** `conda run -n kvgen <command>` OR activate kvgen first
- Examples:
  ```bash
  conda run -n kvgen pip install <package>
  conda run -n kvgen python script.py
  ```
- Or activate first:
  ```bash
  conda activate kvgen
  python script.py
  ```

### Why kvgen?
- Contains custom-built PyTorch 2.10.0a0 with sm_120 support
- Has all project dependencies installed
- Configured with proper DLL paths for CUDA/CUPTI
- Required for GPU inference on RTX 5070 Ti

## Project Structure
- `src/` - Main source code
  - `models/` - Model loading and management
  - `chat/` - Chat interface logic
  - `utils/` - Utility functions
- `config/` - Configuration files
- `examples/` - Example scripts
- `tests/` - Test files
- `fix_dll_paths.py` - **CRITICAL**: Must be imported before torch in all scripts
- `web_app_multi.py` - Multi-model Gradio web interface
- `cli_app.py` - Command-line chat interface

## Development Guidelines
- Use Hugging Face transformers library for model management
- Support multiple model types (DeepSeek Coder, DeepSeek LLM, DialoGPT, etc.)
- Provide both CLI and web interface options
- Follow Python best practices and PEP 8 style guide
- Use type hints for better code clarity
- Handle errors gracefully with informative messages
- **Always import `fix_dll_paths` before importing torch**

## Hardware Setup
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **Compute Capability:** sm_120 (Blackwell architecture)
- **PyTorch:** Custom-built 2.10.0a0 from source
- **CUDA:** v13.0
- **Driver:** 581.42

## Key Files & Their Purpose
- `fix_dll_paths.py` - Sets up CUPTI and CUDA DLL paths (required for PyTorch to load)
- `config/config.yaml` - Model and generation parameters
- `web_app_multi.py` - Multi-model web UI with model switching
- `run_chatbot.bat` - Windows launcher script
- `scripts/build_pytorch_sm120.bat` - PyTorch source build script

## Installation & Dependencies
All packages must be installed in `kvgen` environment:
```bash
conda run -n kvgen pip install <package>
```

Key packages:
- transformers
- torch (custom-built, already in kvgen)
- gradio (for web UI)
- pyyaml, numpy, etc.

## Running the Application

### Web Interface (Recommended)
```bash
conda activate kvgen
python web_app_multi.py
```
Opens at http://127.0.0.1:7860

### CLI Interface
```bash
conda activate kvgen
python cli_app.py
```

### Windows Launcher
Double-click `run_chatbot.bat`

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** You're in the wrong environment. Use `kvgen`:
```bash
conda activate kvgen
```

### Issue: "WinError 126" DLL loading errors
**Solution:** Ensure `fix_dll_paths.py` is imported at the top of the script BEFORE torch

### Issue: "probability tensor contains inf, nan"
**Solution:** Reduce max_length in config.yaml (current: 150, max safe: 256)

## Setup Completed
✓ Project structure created
✓ Core modules implemented  
✓ Configuration files added
✓ Dependencies installed in kvgen
✓ Custom PyTorch built with sm_120 support
✓ DLL path fixes implemented
✓ Multi-model web UI created
✓ GPU inference working (12.56 GB VRAM usage)
