# KVGenius - AI Studio 🎨🤖

A comprehensive AI creative studio combining chat, image generation, and LoRA training. Built for NVIDIA RTX 5070 Ti with custom PyTorch sm_120 (Blackwell) support.

## Features

- 🤖 **Multi-Model Chat** - Switch between multiple LLMs (Mistral, DeepSeek, Dolphin, etc.)
- 🎨 **Image Generation** - Stable Diffusion with multiple models (Dreamshaper, SDXL Turbo, Realistic Vision)
- 🎯 **LoRA Training** - Train custom LoRAs for specific subjects/styles
- ✨ **Prompt Enhancement** - AI-powered prompt improvement (Flan-T5 on CPU)
- 💾 **Prompt Library** - Save and reuse your best prompts
- 📝 **Chat History** - SQLite-backed conversation persistence
- 🎭 **AI Characters & Personas** - Create custom chat personalities

## Project Structure

```
KVGenius/
├── data/                    # User data (gitignored)
│   ├── model_cache/         # Downloaded models
│   ├── lora_models/         # Trained LoRAs
│   ├── training_datasets/   # Training images
│   ├── generated_images/    # Output images
│   └── chat_history.db      # Chat database
│
├── src/
│   ├── chat/                # Chat logic
│   ├── database/            # SQLite management
│   ├── image/               # Image generation module
│   │   ├── generator.py     # SD pipeline wrapper
│   │   ├── enhancer.py      # Prompt enhancement
│   │   └── presets.py       # Model presets
│   ├── models/              # Model loading
│   ├── training/            # LoRA training
│   │   └── lora_trainer.py  # Full training pipeline
│   └── utils/               # Utilities
│
├── scripts/                 # Utility scripts
│   ├── download_model.py
│   ├── download_image_models.py
│   ├── download_civitai_lora.py
│   ├── convert_images_to_png.py
│   └── build_pytorch_sm120.bat
│
├── config/
│   ├── config.yaml          # Main configuration
│   └── image_model_presets.yaml  # Image model settings
│
├── desktop_app.py           # Main Flet desktop application
├── cli_app.py               # CLI interface
├── fix_dll_paths.py         # CUDA DLL path fix (required)
└── requirements.txt
```

## Requirements

### Hardware
- **GPU:** NVIDIA RTX 5070 Ti (16 GB VRAM) or similar
- **CUDA:** 13.0+ for Blackwell/sm_120 support
- **RAM:** 32 GB recommended

### Software
- Python 3.10+
- Miniconda/Anaconda
- Custom PyTorch 2.10+ with sm_120 support (see setup)

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

### 3. Install PyTorch with sm_120 support
For RTX 5070 Ti / Blackwell GPUs, you need custom-built PyTorch:
```powershell
# See scripts/build_pytorch_sm120.bat for source build instructions
# Or install pre-built wheels if available
```

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure (optional)
```powershell
copy .env.example .env
# Edit config/config.yaml as needed
```

## Usage

### Desktop Application (Recommended)
```powershell
# Activate environment
conda activate kvgen

# Run the desktop app
python desktop_app.py
```

### CLI Interface
```powershell
conda activate kvgen
python cli_app.py
```

## Tabs Overview

### 💬 Chat Tab
- Multi-model support with hot-swapping
- AI Characters with custom system prompts
- User Personas for roleplay
- Conversation history with branching

### 🎨 Image Generation Tab
- Multiple SD models with auto-presets
- LoRA support with adjustable strength
- Prompt enhancement (Quick templates or AI-powered)
- Live CLIP token counter (77 token limit)
- Gallery with metadata viewing

### 🎯 LoRA Training Tab
- **Dataset sub-tab:** Upload and caption training images
- **Train sub-tab:** Configure and run training
- **My LoRAs sub-tab:** Manage trained LoRAs
- Multiple crop modes (resize_pad, smart, center, top, stretch)
- Resume training from existing LoRAs

### 📚 Prompt Library Tab
- Save generation settings for reuse
- Organize prompts by category

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

# Inspect model architecture
python scripts/inspect_model.py
```

## Configuration

### config/config.yaml
```yaml
model:
  cache_dir: "./data/model_cache"
  device: "auto"

generation:
  max_length: 150
  temperature: 0.7
  top_p: 0.95
```

### config/image_model_presets.yaml
Per-model defaults for steps, guidance, resolution, etc.

## Supported Models

### Chat Models
| Model | Size | Best For |
|-------|------|----------|
| Nous-Hermes-2-Mistral-7B | 7B | Roleplay, instructions |
| Dolphin-2.6-Mistral-7B | 7B | Uncensored chat |
| DeepSeek Coder 6.7B | 6.7B | Code generation |
| Kunoichi-DPO-v2-7B | 7B | Creative writing |
| DialoGPT Medium | 355M | Fast, lightweight |

### Image Models
| Model | VRAM | Speed |
|-------|------|-------|
| Dreamshaper 8 | ~4 GB | Medium |
| Realistic Vision V5 | ~4 GB | Medium |
| SDXL Turbo | ~7 GB | Fast (1-8 steps) |
| Absolute Reality | ~4 GB | Medium |

## Troubleshooting

### "WinError 126" DLL errors
Ensure `fix_dll_paths.py` is imported before torch in all scripts.

### CUDA/sm_120 not supported
You need custom-built PyTorch. See `scripts/build_pytorch_sm120.bat`.

### Out of VRAM
- Unload unused models (auto-unload is enabled)
- Use smaller models
- Reduce image resolution

### Black images from SD
Safety checker is disabled by default. If still getting black images, try different seeds or reduce guidance scale.

## License

Open source for educational purposes.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Flet](https://flet.dev/) for desktop UI
- [PEFT](https://github.com/huggingface/peft) for LoRA training

---

**Built for RTX 5070 Ti with ❤️**
