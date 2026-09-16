# KVGenius - AI Studio 🎨🤖

A comprehensive AI creative studio combining chat, image generation, and LoRA training. Chat runs against a local **Ollama** server; image generation runs against a local **ComfyUI** server — KVGenius is a Flet desktop client for both, not an in-process model runner.

This repo follows the shared conventions in
[gerp93/KVG_Standards](https://github.com/gerp93/KVG_Standards) — theming,
release/CI, self-update, licensing, and database location all come from
there rather than being reinvented locally.

## Features

- 🤖 **Multi-Model Chat** - Switch between any model pulled into Ollama (Mistral, DeepSeek, Dolphin, etc.)
- 🎨 **Image Generation** - Any checkpoint ComfyUI can see, with LoRA support
- 🎯 **Image LoRA Training** - Train custom image LoRAs for specific subjects/styles
- 💾 **Prompt Library** - Save and reuse your best prompts
- 📝 **Chat History** - SQLite-backed conversation persistence
- 🎭 **AI Characters & Personas** - Create custom chat personalities
- 🃏 **Card Generator** - Generate CAH-style party game cards from a topic or an article

## Project Structure

```
KVGenius/
├── data/                    # User data (gitignored)
│   ├── checkpoints/         # Downloaded image checkpoints
│   ├── lora_models/         # Image LoRA files
│   ├── training_datasets/   # Training images
│   ├── generated_images/    # Output images
│   └── chat_history.db      # Chat database
│
├── core/                    # UI-agnostic backend
│   ├── chat_gen.py          # Ollama client wrapper
│   ├── image_gen.py         # ComfyUI client wrapper
│   ├── ollama_client.py     # Thin Ollama REST client
│   ├── comfyui_client.py    # Thin ComfyUI REST client
│   └── ...                  # config, prompt_builder, semantic_index, db_location
│
├── src/
│   ├── database/             # SQLite management
│   ├── training/
│   │   └── lora_trainer.py  # Image LoRA training pipeline (local torch/diffusers)
│   └── cards/                # Card generator logic
│
├── ui/tabs/card_generator.py  # The one ui/tabs/* module wired into desktop_app.py
│                                 (most other tab UIs are implemented inline in desktop_app.py)
│
├── scripts/                 # Utility scripts
│   ├── download_model.py
│   ├── download_image_models.py
│   ├── download_civitai_lora.py
│   ├── convert_images_to_png.py
│   └── build_pytorch_sm120.bat
│
├── config/
│   ├── settings.yaml               # ollama_host / comfyui_host and other settings
│   ├── chat_model_presets.yaml     # Ollama model tags
│   ├── image_model_presets.yaml    # ComfyUI checkpoint filenames
│   └── comfyui_workflows/          # ComfyUI workflow JSON templates
│
├── desktop_app.py           # Main Flet desktop application (only supported entry point)
├── fix_dll_paths.py         # CUDA DLL path fix (needed only for LoRA training / semantic search)
└── requirements.txt
```

## Requirements

### Software
- Python 3.10+
- [Ollama](https://ollama.com/) running locally, with your chat models pulled (`ollama pull mistral:7b`, etc.)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally, with your checkpoints in its checkpoints folder
- Miniconda/Anaconda (recommended)

### Optional (only for local image-LoRA training / semantic memory search)
- NVIDIA GPU + PyTorch with matching CUDA/sm support — `src/training/lora_trainer.py` and `core/semantic_index.py` still run local `torch`/`diffusers`/`peft` for these two features. Chat and image generation themselves don't need a local GPU stack at all.

## Installation

### 1. Clone the repository
```powershell
git clone https://github.com/gerp93/KVGenius.git
cd KVGenius
```

### 2. Create conda environment
```powershell
conda create -n kvgen python=3.10
conda activate kvgen
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure
```powershell
copy config\chat_model_presets.example.yaml config\chat_model_presets.yaml
copy config\image_model_presets.example.yaml config\image_model_presets.yaml
# Edit ollama_host / comfyui_host in config/settings.yaml if they're not on localhost defaults
```

### 5. Start Ollama and ComfyUI
KVGenius does not launch these for you - start them yourself before using the Chat or Image Generation tabs:
```powershell
ollama serve
# and, in another terminal, start ComfyUI per its own instructions
```

## Usage

### Desktop Application (only supported entry point)
```powershell
# Activate environment
conda activate kvgen

# Run the desktop app
python desktop_app.py
```

## Tabs Overview

### 💬 Chat Tab
- Multi-model support with hot-swapping (any model pulled into Ollama)
- AI Characters with custom system prompts
- User Personas for roleplay
- Conversation history with branching

### 🎨 Image Generation Tab
- Any ComfyUI-visible checkpoint, with auto-presets
- LoRA support with adjustable strength
- Live CLIP token counter (77 token limit)
- Gallery with metadata viewing

### 🎯 LoRA Training Tab
- **Dataset sub-tab:** Upload and caption training images
- **Train sub-tab:** Configure and run image LoRA training
- **My LoRAs sub-tab:** Manage trained LoRAs
- Multiple crop modes (resize_pad, smart, center, top, stretch)
- Resume training from existing LoRAs

### 📚 Prompt Library Tab
- Save generation settings for reuse
- Organize prompts by category

### 🃏 Card Generator Tab
- Generate CAH-style black/white cards from a topic, or extract them from an article
- Card library with search, filtering, favorites, and export

## Utility Scripts

```powershell
# Download specific model
python scripts/download_model.py

# Download all image generation models
python scripts/download_image_models.py

# Download LoRA from CivitAI
python scripts/download_civitai_lora.py <url>

# Convert training images to PNG
python scripts/convert_images_to_png.py [folder]
```

## Configuration

### config/settings.yaml
```yaml
ollama_host: "http://localhost:11434"
comfyui_host: "http://localhost:8188"
```

### config/chat_model_presets.yaml
Ollama model tags (must already be `ollama pull`-ed) plus display metadata.

### config/image_model_presets.yaml
Per-model ComfyUI checkpoint filenames and defaults for steps, guidance, resolution, etc.

### config/comfyui_workflows/
ComfyUI workflow JSON templates (one per model family: SD1.5, SDXL, Flux) that `core/comfyui_client.py` patches per request. See `config/comfyui_workflows/README.md`.

## Supported Models

Example presets ship in `config/chat_model_presets.example.yaml` and
`config/image_model_presets.example.yaml` — copy them to the non-`.example`
filename and add/replace entries with whatever you've pulled into Ollama or
placed in ComfyUI's checkpoints folder. Nothing is hardcoded beyond those
config files.

## Troubleshooting

### "Ollama not reachable" / "ComfyUI not reachable"
Start the relevant service before generating, and confirm `ollama_host`/`comfyui_host` in `config/settings.yaml` match where it's actually listening.

### "WinError 126" DLL errors (LoRA training only)
Ensure `fix_dll_paths.py` is imported before torch in any script that touches it.

### CUDA/sm_120 not supported (LoRA training only)
You need a PyTorch build matching your GPU's compute capability. See `scripts/build_pytorch_sm120.bat`.

## License

[AGPL-3.0](LICENSE) — see [gerp93/KVG_Standards' licensing.md](https://github.com/gerp93/KVG_Standards/blob/main/licensing.md) for the org-wide default this repo follows.

## Acknowledgments

- [Ollama](https://ollama.com/) for local chat model serving
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for local image generation
- [Flet](https://flet.dev/) for desktop UI
- [PEFT](https://github.com/huggingface/peft) / [Diffusers](https://github.com/huggingface/diffusers) for image LoRA training

---

**Built with ❤️**
