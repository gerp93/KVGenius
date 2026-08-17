# KVGenius - Project Instructions

## Project Overview
KVGenius is a Flet-based desktop AI studio: chat, image generation, and LoRA training in one app. Chat generation runs against a local **Ollama** server; image generation runs against a local **ComfyUI** server. Neither model backend is loaded in-process by KVGenius itself — the app is an HTTP client to both services.

## CRITICAL: Environment Setup
**⚠️ ALWAYS USE THE `kvgen` CONDA ENVIRONMENT FOR ALL OPERATIONS**

### Running Commands
- **NEVER run commands in `base` environment**
- **ALWAYS use:** `conda run -n kvgen <command>` OR activate kvgen first
  ```bash
  conda run -n kvgen pip install <package>
  conda run -n kvgen python script.py
  ```
  or
  ```bash
  conda activate kvgen
  python script.py
  ```

### External services required at runtime
- **Ollama** must be running locally (default `http://localhost:11434`) for the Chat and Card Generator tabs to work.
- **ComfyUI** must be running locally (default `http://localhost:8188`) for the Image Generation tab to work.
- KVGenius does not launch either service itself — start them separately before using those tabs. Host/port are configurable in `config/settings.yaml`.

## Project Structure
- `desktop_app.py` — main Flet application entry point (`python desktop_app.py`); most tab UIs are implemented inline in this file.
- `core/` — UI-agnostic backend: `chat_gen.py` (Ollama client wrapper), `image_gen.py` (ComfyUI client wrapper), `ollama_client.py`, `comfyui_client.py`, `config.py`, `prompt_builder.py`, `semantic_index.py`, `db_location.py`.
- `ui/tabs/card_generator.py` — the one `ui/tabs/*` module actually wired into `desktop_app.py`; other files under `ui/` are not currently imported by the running app.
- `src/database/` — SQLite chat history.
- `src/training/lora_trainer.py` — image LoRA training pipeline (local `torch`/`diffusers`/`peft`, still in-process — this is the one place those heavy ML packages are still required).
- `src/cards/` — Cards Against Humanity-style card generator logic.
- `config/*.yaml` — model presets, settings, prompt templates.
- `fix_dll_paths.py` — **CRITICAL**: Must be imported before `torch` in any script that imports `torch` (still needed for `src/training/lora_trainer.py` and `core/semantic_index.py`'s local embedding model).
- `scripts/` — utility scripts (model/LoRA downloads, image conversion, PyTorch build helper).

## Development Guidelines
- Chat and image generation go through `core/ollama_client.py` / `core/comfyui_client.py` — do not add new in-process `transformers`/`diffusers` model loading for those two features.
- Follow Python best practices and PEP 8 style guide.
- Use type hints for better code clarity.
- Handle service-unavailable errors from Ollama/ComfyUI gracefully — surface a clear message in the UI rather than a raw exception/traceback.
- **Always import `fix_dll_paths` before importing `torch`** in any script/module that touches `torch` (LoRA training, semantic index).

## Hardware Setup
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **Compute Capability:** sm_120 (Blackwell architecture)
- **PyTorch:** Custom-built from source (see `scripts/build_pytorch_sm120.bat`) — only needed for local LoRA training / semantic search, not for chat or image generation anymore.
- **CUDA:** v13.0+

## Running the Application

### Desktop app (only supported entry point)
```bash
conda activate kvgen
python desktop_app.py
```

## Common Issues & Solutions

### Issue: "Ollama not reachable" / "ComfyUI not reachable"
**Solution:** Start the relevant service before generating (`ollama serve`, or launch ComfyUI), and confirm the host/port in `config/settings.yaml` match where it's actually listening.

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** You're in the wrong environment. Use `kvgen`:
```bash
conda activate kvgen
```

### Issue: "WinError 126" DLL loading errors
**Solution:** Ensure `fix_dll_paths.py` is imported at the top of the script before `torch`.
