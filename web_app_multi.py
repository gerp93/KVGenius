"""
Multi-Model Web interface for the AI chatbot using Gradio.
"""
# CRITICAL: Import DLL path fix BEFORE any torch imports
import fix_dll_paths

import sys
import os
import time
import yaml
import json
import requests
from bs4 import BeautifulSoup

# Force unbuffered output for conda run
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

print("[KVGenius] Starting up...", flush=True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelLoader
from src.chat import ChatBot
from src.utils import load_config, setup_logging, load_environment
from src.database import ChatHistoryDB
from src.training.lora_trainer import (
    get_dataset_manager, get_lora_manager, get_trainer,
    DatasetManager, LoRAManager, LoRATrainer, TrainingConfig, training_state
)
from src.cards.cah_generator import (
    CAHGenerator, CardType, CardStyle, 
    get_generation_prompt, get_article_extraction_prompt, ARTICLE_EXTRACTION_PROMPT
)
import gradio as gr
import logging
import torch

# Configure logging to show in console with unbuffered handler
class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[FlushHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
print("[KVGenius] Logging configured", flush=True)

# Global variables
active_models = {}
gen_config = {}
cache_dir = "./data/model_cache"
is_downloading = False
download_canceled = False
db = None  # Database instance
current_conversation_id = None  # Active conversation ID
image_model = None  # Stable Diffusion pipeline
image_model_loaded = False
loaded_image_models = {}  # cache of loaded image pipelines by name
current_image_model_key = None  # track which model is currently active
image_gen_cancel_event = False  # flag to request cancellation of a running generation
image_gen_in_progress = False

# Cards Against Humanity generator instance
cah_generator = None


# CPU Prompt Enhancer (lazy loaded)
prompt_enhancer_model = None
prompt_enhancer_tokenizer = None

# CLIP token limit - applies to all SD/SDXL models (hardcoded in CLIP architecture)
CLIP_MAX_TOKENS = 77


def get_gpu_info():
    """Get GPU name and compute capability for display."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Get compute capability
        major, minor = torch.cuda.get_device_capability(0)
        return f"{gpu_name} (sm_{major}{minor})"
    else:
        return "CPU (No GPU detected)"


def count_clip_tokens(text, tokenizer=None):
    """Count actual CLIP tokens using the model's tokenizer, or estimate if not available."""
    if not text:
        return 0
    
    # Try to use the actual tokenizer from loaded model
    if tokenizer is not None:
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            return tokens.input_ids.shape[1]
        except:
            pass
    
    # Fallback: estimate based on CLIP tokenization patterns
    # CLIP uses BPE - roughly 1.3 tokens per word on average
    import re
    words = len(re.findall(r'\w+', text))
    punctuation = len(re.findall(r'[^\w\s]', text))
    # Add buffer for subword tokenization
    return int(words * 1.3) + punctuation


def update_token_counter(prompt):
    """Update the live token counter display."""
    if not prompt or not prompt.strip():
        return "*0 / 77 tokens*"
    
    # Use actual tokenizer if model is loaded
    tokenizer = getattr(image_model, 'tokenizer', None) if image_model else None
    tokens = count_clip_tokens(prompt, tokenizer)
    
    if tokens > CLIP_MAX_TOKENS:
        return f"### ⚠️ {tokens} / 77 TOKENS - PROMPT WILL BE TRUNCATED!"
    elif tokens > 60:
        return f"**{tokens} / 77 tokens** ⚠️ *getting close to limit*"
    else:
        return f"*{tokens} / 77 tokens*"


def read_image_metadata(filepath):
    """Read generation metadata from a PNG file."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        metadata = {}
        if hasattr(img, 'info'):
            for key in ['prompt', 'negative_prompt', 'steps', 'guidance_scale', 'width', 'height', 'seed', 'model', 'generated_at']:
                if key in img.info:
                    metadata[key] = img.info[key]
        return metadata
    except Exception as e:
        logger.error(f"Error reading metadata from {filepath}: {e}")
        return {}


def format_image_metadata(metadata):
    """Format metadata for display."""
    if not metadata:
        return "*No metadata available*"
    
    lines = []
    if metadata.get('prompt'):
        lines.append(f"**Prompt:** {metadata['prompt']}")
    if metadata.get('negative_prompt'):
        lines.append(f"**Negative:** {metadata['negative_prompt']}")
    if metadata.get('model'):
        lines.append(f"**Model:** {metadata['model']}")
    
    settings = []
    if metadata.get('steps'):
        settings.append(f"Steps: {metadata['steps']}")
    if metadata.get('guidance_scale'):
        settings.append(f"CFG: {metadata['guidance_scale']}")
    if metadata.get('width') and metadata.get('height'):
        settings.append(f"Size: {metadata['width']}x{metadata['height']}")
    if metadata.get('seed'):
        settings.append(f"Seed: {metadata['seed']}")
    
    if settings:
        lines.append(f"**Settings:** {' | '.join(settings)}")
    
    if metadata.get('generated_at'):
        lines.append(f"**Generated:** {metadata['generated_at']}")
    
    return "\n\n".join(lines)


def load_utility_model_presets():
    """Load utility model presets from config file."""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    presets_path = os.path.join(config_dir, "utility_model_presets.yaml")
    
    try:
        with open(presets_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('models', {})
    except Exception as e:
        logger.warning(f"Could not load utility model presets: {e}")
        # Return defaults
        return {
            'prompt_enhancer': {'id': 'google/flan-t5-small', 'device': 'cpu'},
            'image_captioner': {'id': 'Salesforce/blip-image-captioning-base', 'device': 'cpu'},
            'nsfw_detector': {'id': 'Falconsai/nsfw_image_detection', 'device': 'cpu'}
        }


# Load utility model presets from config
UTILITY_MODELS = load_utility_model_presets()


def load_image_model_presets():
    """Load image model presets from config file.
    
    If user config doesn't exist, copies from example template.
    Returns both HuggingFace models and local checkpoints.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    presets_path = os.path.join(config_dir, "image_model_presets.yaml")
    example_path = os.path.join(config_dir, "image_model_presets.example.yaml")
    
    # If user config doesn't exist, copy from example
    if not os.path.exists(presets_path) and os.path.exists(example_path):
        import shutil
        shutil.copy(example_path, presets_path)
        logger.info(f"Created image_model_presets.yaml from example template")
    
    try:
        with open(presets_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            models = config.get('models', {})
            checkpoints = config.get('checkpoints', {})
            prompt_templates = config.get('prompt_templates', {})
            return models, checkpoints, prompt_templates
    except Exception as e:
        logger.warning(f"Could not load image model presets: {e}")
        return {}, {}, {}


# Load image model presets from config
IMAGE_MODEL_PRESETS, CHECKPOINT_PRESETS, PROMPT_TEMPLATES = load_image_model_presets()

# Build AVAILABLE_IMAGE_MODELS from presets (with fallback defaults)
AVAILABLE_IMAGE_MODELS = {}
for model_key, preset in IMAGE_MODEL_PRESETS.items():
    AVAILABLE_IMAGE_MODELS[model_key] = {
        "id": preset.get("id", ""),
        "description": preset.get("description", ""),
        "vram": preset.get("vram", "~6 GB"),
        "default_steps": preset.get("default_steps", 25),
        "default_guidance": preset.get("default_guidance", 7.5),
        "default_width": preset.get("default_width", 512),
        "default_height": preset.get("default_height", 512),
        "step_range": preset.get("step_range", [1, 50]),
        "guidance_range": preset.get("guidance_range", [0.0, 20.0]),
        "default_negative": preset.get("default_negative", ""),
        "tips": preset.get("tips", ""),
        "is_checkpoint": False,
    }

# Add local checkpoints to available models
checkpoints_dir = os.path.join(os.path.dirname(__file__), "data", "checkpoints")
if CHECKPOINT_PRESETS:
    for model_key, preset in CHECKPOINT_PRESETS.items():
        checkpoint_path = preset.get("path", "")
        full_path = os.path.join(checkpoints_dir, checkpoint_path)
        if os.path.exists(full_path):
            AVAILABLE_IMAGE_MODELS[f"📁 {model_key}"] = {
                "id": full_path,  # Full path for checkpoint loading
                "description": preset.get("description", "Local checkpoint"),
                "vram": preset.get("vram", "~4 GB"),
                "default_steps": preset.get("default_steps", 25),
                "default_guidance": preset.get("default_guidance", 7.0),
                "default_width": preset.get("default_width", 512),
                "default_height": preset.get("default_height", 768),
                "step_range": preset.get("step_range", [15, 50]),
                "guidance_range": preset.get("guidance_range", [5.0, 12.0]),
                "default_negative": preset.get("default_negative", ""),
                "tips": preset.get("tips", ""),
                "is_checkpoint": True,
                "checkpoint_type": preset.get("type", "sd15"),
            }
            logger.info(f"Added checkpoint: {model_key} from {checkpoint_path}")

# Also auto-discover checkpoints not in config
if os.path.exists(checkpoints_dir):
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith(('.safetensors', '.ckpt')) and not filename.startswith('.'):
            model_name = os.path.splitext(filename)[0]
            display_name = f"📁 {model_name}"
            if display_name not in AVAILABLE_IMAGE_MODELS:
                full_path = os.path.join(checkpoints_dir, filename)
                file_size_gb = os.path.getsize(full_path) / (1024**3)
                is_sdxl = file_size_gb > 4.0 or "xl" in filename.lower() or "sdxl" in filename.lower()
                AVAILABLE_IMAGE_MODELS[display_name] = {
                    "id": full_path,
                    "description": f"Local checkpoint ({file_size_gb:.1f}GB)",
                    "vram": "~8 GB" if is_sdxl else "~4 GB",
                    "default_steps": 25,
                    "default_guidance": 7.0,
                    "default_width": 1024 if is_sdxl else 512,
                    "default_height": 1024 if is_sdxl else 768,
                    "step_range": [15, 50],
                    "guidance_range": [5.0, 12.0],
                    "default_negative": "ugly, deformed, blurry, low quality",
                    "tips": f"Auto-detected {'SDXL' if is_sdxl else 'SD 1.5'} checkpoint",
                    "is_checkpoint": True,
                    "checkpoint_type": "sdxl" if is_sdxl else "sd15",
                }
                logger.info(f"Auto-discovered checkpoint: {model_name} ({'SDXL' if is_sdxl else 'SD15'})")

# Fallback if config file not found
if not AVAILABLE_IMAGE_MODELS:
    AVAILABLE_IMAGE_MODELS = {
        "Dreamshaper 8": {
            "id": "Lykon/dreamshaper-8",
            "description": "Artistic, great for portraits",
            "vram": "~4 GB",
            "default_steps": 25,
            "default_guidance": 7.5,
            "default_width": 512,
            "default_height": 768,
            "step_range": [15, 50],
            "guidance_range": [5.0, 12.0],
            "default_negative": "ugly, deformed, blurry",
            "tips": ""
        },
        "SDXL Turbo": {
            "id": "stabilityai/sdxl-turbo",
            "description": "Fast generation (1-8 steps)",
            "vram": "~7 GB",
            "default_steps": 6,
            "default_guidance": 0.5,
            "default_width": 1024,
            "default_height": 1024,
            "step_range": [1, 8],
            "guidance_range": [0.0, 2.0],
            "default_negative": "ugly, deformed, blurry",
            "tips": ""
        }
    }


def load_chat_model_presets():
    """Load chat model presets from config file.
    
    If user config doesn't exist, copies from example template.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    presets_path = os.path.join(config_dir, "chat_model_presets.yaml")
    example_path = os.path.join(config_dir, "chat_model_presets.example.yaml")
    
    # If user config doesn't exist, copy from example
    if not os.path.exists(presets_path) and os.path.exists(example_path):
        import shutil
        shutil.copy(example_path, presets_path)
        logger.info(f"Created chat_model_presets.yaml from example template")
    
    try:
        with open(presets_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('models', {})
    except Exception as e:
        logger.warning(f"Could not load chat model presets: {e}")
        return {}


# Load chat model presets from config
CHAT_MODEL_PRESETS = load_chat_model_presets()

# Build AVAILABLE_MODELS from presets (with fallback defaults)
AVAILABLE_MODELS = {}
for model_key, preset in CHAT_MODEL_PRESETS.items():
    AVAILABLE_MODELS[model_key] = {
        "id": preset.get("id", ""),
        "description": preset.get("description", ""),
        "details": preset.get("details", ""),
        "best_for": preset.get("best_for", ""),
        "params": preset.get("params", ""),
        "vram": preset.get("vram", "~14 GB")
    }

# Fallback if config file not found
if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = {
        "DialoGPT Medium": {
            "id": "microsoft/DialoGPT-medium",
            "description": "⚡ Fast & lightweight",
            "details": "Smallest and fastest option for quick responses.",
            "best_for": "Quick responses, low VRAM usage",
            "params": "355M",
            "vram": "~2 GB"
        }
    }


# ============================================================
# MODEL STATUS HELPERS
# ============================================================

def is_model_downloaded(repo_id: str, is_image_model: bool = False) -> bool:
    """Check if a model is downloaded to disk.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'Lykon/dreamshaper-8')
        is_image_model: If True, check for complete SD model (unet + vae)
    """
    # HuggingFace cache structure: models--org--name
    cache_name = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(cache_dir, cache_name)
    
    if not os.path.exists(cache_path):
        return False
    
    # Check for snapshots folder with actual model files
    snapshots_path = os.path.join(cache_path, "snapshots")
    if os.path.exists(snapshots_path):
        for snapshot in os.listdir(snapshots_path):
            snapshot_dir = os.path.join(snapshots_path, snapshot)
            if os.path.isdir(snapshot_dir):
                if is_image_model:
                    # For SD models, check that unet is fully downloaded
                    unet_path = os.path.join(snapshot_dir, "unet")
                    if os.path.isdir(unet_path):
                        unet_files = [f for f in os.listdir(unet_path) 
                                      if f.endswith(('.safetensors', '.bin'))]
                        if unet_files:
                            # Check file isn't being downloaded (has .incomplete suffix or is very small)
                            for uf in unet_files:
                                full_path = os.path.join(unet_path, uf)
                                # UNet should be at least 1GB for SD models
                                if os.path.getsize(full_path) > 1_000_000_000:
                                    return True
                    return False
                else:
                    # For chat models, any model file indicates downloaded
                    for root, dirs, files in os.walk(snapshot_dir):
                        for f in files:
                            if f.endswith(('.safetensors', '.bin', '.pt')):
                                return True
    return False


def get_chat_model_status(model_key: str) -> str:
    """Get status icon for a chat model: 🟢 Loaded | 🟡 Downloaded | 🔴 Not Downloaded"""
    global active_models
    
    if model_key in active_models:
        return "🟢"  # Loaded in VRAM
    
    if model_key in AVAILABLE_MODELS:
        repo_id = AVAILABLE_MODELS[model_key]["id"]
        if is_model_downloaded(repo_id):
            return "🟡"  # Downloaded but not loaded
    
    return "🔴"  # Not downloaded


def get_image_model_status(model_key: str) -> str:
    """Get status icon for an image model: 🟢 Loaded | 🟡 Downloaded | 🔴 Not Downloaded"""
    global loaded_image_models
    
    if model_key in loaded_image_models:
        return "🟢"  # Loaded in VRAM
    
    if model_key in AVAILABLE_IMAGE_MODELS:
        repo_id = AVAILABLE_IMAGE_MODELS[model_key]["id"]
        if is_model_downloaded(repo_id, is_image_model=True):
            return "🟡"  # Downloaded but not loaded
    
    return "🔴"  # Not downloaded


def get_chat_model_choices():
    """Get chat model choices with status icons."""
    choices = []
    for model_key in AVAILABLE_MODELS.keys():
        status = get_chat_model_status(model_key)
        choices.append(f"{status} {model_key}")
    return choices


def get_image_model_choices():
    """Get image model choices with status icons."""
    choices = []
    for model_key in AVAILABLE_IMAGE_MODELS.keys():
        status = get_image_model_status(model_key)
        choices.append(f"{status} {model_key}")
    return choices


def refresh_image_dropdown(current_selection):
    """Refresh image dropdown choices and update selected value to match new status."""
    choices = get_image_model_choices()
    # Extract the model key from current selection and find matching choice
    if current_selection:
        model_key = extract_model_key(current_selection)
        for choice in choices:
            if model_key in choice:
                return gr.update(choices=choices, value=choice)
    return gr.update(choices=choices)


def refresh_chat_dropdown(current_selection):
    """Refresh chat dropdown choices and update selected value to match new status."""
    choices = get_chat_model_choices()
    if current_selection:
        model_key = extract_model_key(current_selection)
        for choice in choices:
            if model_key in choice:
                return gr.update(choices=choices, value=choice)
    return gr.update(choices=choices)


def extract_model_key(selection: str) -> str:
    """Extract model key from selection with status icon (e.g., '🟢 ModelName' -> 'ModelName')."""
    if selection and len(selection) > 2:
        # Remove status icon prefix (emoji + space)
        if selection[0] in "🟢🟡🔴":
            return selection[2:].strip()
    return selection


def unload_all_chat_models():
    """Unload all cached chat models to free VRAM."""
    global active_models
    
    for model_key in list(active_models.keys()):
        try:
            model, tokenizer, chatbot = active_models[model_key]
            del model
            del tokenizer
            del chatbot
            logger.info(f"Unloaded chat model: {model_key}")
        except:
            pass
    
    active_models.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_key, progress=gr.Progress()):
    """Load a model if not cached. Auto-unloads previous model to save VRAM."""
    global is_downloading, download_canceled, active_models
    
    # If this model is already loaded, return it
    if model_key in active_models:
        return active_models[model_key]
    
    # Auto-unload any previously loaded models to free VRAM
    if active_models:
        prev_models = list(active_models.keys())
        progress(0.05, desc="🗑️ Unloading previous model to free VRAM...")
        unload_all_chat_models()
        logger.info(f"Auto-unloaded previous chat models: {prev_models}")
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]
    
    logger.info(f"Loading {model_key}...")
    
    # Check if model is cached
    model_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    is_cached = os.path.exists(model_path)
    
    if not is_cached:
        is_downloading = True
        download_canceled = False
        progress(0.1, desc=f"📥 Downloading {model_key}... (this may take several minutes)")
    else:
        progress(0.1, desc=f"📦 Found {model_key} in cache, loading...")
    
    try:
        progress(0.15, desc="⚙️ Initializing model loader...")
        model_loader = ModelLoader(
            model_name=model_id,
            cache_dir=cache_dir,
            device="auto",
            token=None
        )
        
        # Simulate progress updates (Hugging Face downloads happen in background)
        if not is_cached:
            progress(0.2, desc=f"📥 Downloading {model_key}... Please wait...")
            for i in range(5):
                if download_canceled:
                    is_downloading = False
                    raise Exception("Download canceled by user")
                progress(0.2 + (i * 0.1), desc=f"📥 Downloading {model_key}... {20 + (i*15)}%")
                import time
                time.sleep(0.5)
        
        progress(0.7, desc="📦 Loading model weights...")
        model, tokenizer = model_loader.load()
        
        progress(0.85, desc="🤖 Setting up chatbot...")
        chatbot = ChatBot(
            model=model,
            tokenizer=tokenizer,
            device=model_loader.device,
            max_history=5
        )
        
        progress(0.95, desc="✨ Finalizing...")
        active_models[model_key] = (model, tokenizer, chatbot)
        logger.info(f"✓ {model_key} loaded")
        
        is_downloading = False
        progress(1.0, desc=f"✅ {model_key} ready!")
        
        return active_models[model_key]
    
    except Exception as e:
        is_downloading = False
        raise e


def format_prompt_for_model(tokenizer, model_key, system_instruction, char_name, user_persona_context, recent_history, current_message):
    """
    Format prompt using model-specific templates.
    Tries tokenizer's chat template first, then falls back to model-specific formats.
    """
    
    # Build conversation messages
    messages = []
    
    # Add system message if present
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    # Add user persona context if present
    if user_persona_context:
        if messages and messages[-1]["role"] == "system":
            # Append to system message
            messages[-1]["content"] += f"\n\n{user_persona_context}"
        else:
            messages.append({"role": "system", "content": user_persona_context})
    
    # Add conversation history
    for user_msg, bot_msg in recent_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    # DISABLED: Tokenizer chat templates are unreliable for multi-turn conversations
    # Always use our custom formats for better control
    # if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
    #     try:
    #         prompt = tokenizer.apply_chat_template(
    #             messages, 
    #             tokenize=False, 
    #             add_generation_prompt=True
    #         )
    #         logger.info(f"Using tokenizer chat template for {model_key}")
    #         return prompt
    #     except Exception as e:
    #         logger.warning(f"Chat template failed for {model_key}: {e}. Using fallback.")
    
    # Use model-specific formats for reliable multi-turn conversations
    logger.info(f"Using custom prompt format for {model_key}")
    
    # Mistral/Hermes/Kunoichi/Dolphin-Mistral format: [INST] ... [/INST]
    # Note: Dolphin-2.6-Mistral uses Mistral format, not ChatML
    if "Mistral" in model_key or "Hermes" in model_key or "Kunoichi" in model_key or "Dolphin" in model_key:
        prompt_parts = []
        
        # Add system instruction in its own [INST] block
        if system_instruction:
            prompt_parts.append(f"[INST] {system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append(" [/INST] Understood.</s>")  # Close system instruction
        
        # Add conversation history - each gets its own [INST] block
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"[INST] {user_msg} [/INST] {bot_msg}</s>")
        
        # Add current message
        prompt_parts.append(f"[INST] {current_message} [/INST]")
        
        return "".join(prompt_parts)
    
    # Vicuna format: USER: ... ASSISTANT:
    elif "Vicuna" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(system_instruction)
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("\n\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"USER: {user_msg}\nASSISTANT: {bot_msg}\n")
        
        prompt_parts.append(f"USER: {current_message}\nASSISTANT:")
        return "".join(prompt_parts)
    
    # Llama-3 format (used by Dark Champion MOE)
    elif "Llama" in model_key or "Champion" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("<|eot_id|>")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>")
            prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>")
        
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{current_message}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)
    
    # DeepSeek format (ChatML-like)
    elif "DeepSeek" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"<｜begin▁of▁sentence｜>System: {system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}\n")
        
        prompt_parts.append(f"User: {current_message}\nAssistant:")
        return "".join(prompt_parts)
    
    # DialoGPT and generic fallback: simple format
    else:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"{system_instruction}\n\n")
            if user_persona_context:
                prompt_parts.append(f"{user_persona_context}\n\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}\n")
        
        prompt_parts.append(f"User: {current_message}\nAssistant:")
        return "".join(prompt_parts)


def get_status():
    """Get system status."""
    lines = []
    
    if is_downloading:
        lines.append("📥 **Status:** Downloading model...")
        lines.append("")
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"🎮 **GPU:** {gpu}")
        lines.append(f"📊 **VRAM:** {vram_used:.2f} / {vram_total:.1f} GB")
    else:
        lines.append("💻 **Device:** CPU")
    
    if active_models:
        lines.append(f"\n✅ **Loaded:** {len(active_models)} model(s)")
        for name in active_models.keys():
            lines.append(f"  • {name}")
    else:
        lines.append("\n📦 **No models loaded**")
    
    return "\n".join(lines)


# ============================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================

def unload_all_image_models():
    """Unload all cached image models to free VRAM."""
    global image_model, image_model_loaded, loaded_image_models
    
    for model_key in list(loaded_image_models.keys()):
        try:
            del loaded_image_models[model_key]
            logger.info(f"Unloaded image model: {model_key}")
        except:
            pass
    
    loaded_image_models.clear()
    image_model = None
    image_model_loaded = False
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image_model(model_name, progress=gr.Progress()):
    """Load a Stable Diffusion model. Auto-unloads previous model to save VRAM."""
    global image_model, image_model_loaded, loaded_image_models, current_image_model_key
    
    # Extract model key from selection (remove status icon)
    model_key = extract_model_key(model_name)
    
    logger.info(f"load_image_model called with model_name={model_name}, model_key={model_key}")
    logger.info(f"progress type: {type(progress)}")
    
    # Handle case where progress isn't callable (when called from .then() chain)
    def safe_progress(value, desc=""):
        try:
            if callable(progress):
                progress(value, desc=desc)
        except Exception as e:
            logger.debug(f"Progress update skipped: {e}")
    
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
        
        if model_key not in AVAILABLE_IMAGE_MODELS:
            return f"### ❌ Unknown model: {model_key}"
        
        model_id = AVAILABLE_IMAGE_MODELS[model_key]["id"]
        
        safe_progress(0.05, desc="🔍 Checking model...")
        logger.info(f"Loading image model: {model_id}")
        
        # Check if already downloaded
        model_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
        is_cached = os.path.exists(model_path)
        
        # Auto-unload any previously loaded model to free VRAM
        if loaded_image_models:
            prev_models = list(loaded_image_models.keys())
            safe_progress(0.1, desc="🗑️ Unloading previous model to free VRAM...")
            unload_all_image_models()
            logger.info(f"Auto-unloaded previous models: {prev_models}")
        
        if is_cached:
            safe_progress(0.15, desc=f"📦 Found {model_key} in cache, loading...")
        else:
            safe_progress(0.15, desc=f"📥 Downloading {model_key}... (this may take a few minutes)")
        
        # Load appropriate pipeline based on model
        # Check for SDXL models by name hints, or auto-detect from model config
        is_sdxl = "sdxl" in model_id.lower() or "xl" in model_id.lower() or "turbo" in model_key.lower()
        
        # Also check if model has SDXL structure (text_encoder_2 folder)
        if not is_sdxl:
            try:
                from huggingface_hub import hf_hub_download
                import json
                safe_progress(0.2, desc="🔍 Detecting model type...")
                config_path = hf_hub_download(model_id, "model_index.json", cache_dir=cache_dir)
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if config.get('_class_name') == 'StableDiffusionXLPipeline' or 'text_encoder_2' in config:
                    is_sdxl = True
                    logger.info(f"Auto-detected SDXL model from config: {model_id}")
            except:
                pass  # Fall back to name-based detection
        
        safe_progress(0.25, desc=f"⚙️ Loading {'SDXL' if is_sdxl else 'SD 1.5'} pipeline...")
        
        if is_sdxl:
            # Try StableDiffusionXLPipeline first for proper SDXL support
            try:
                from diffusers import EulerDiscreteScheduler
                
                safe_progress(0.3, desc="📥 Loading model weights...")
                image_model = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=cache_dir,
                    use_safetensors=True,
                )
                
                safe_progress(0.6, desc="⚙️ Configuring scheduler...")
                # Some models use EDM schedulers that may cause noise output
                # Replace with a standard Euler scheduler for compatibility
                scheduler_type = type(image_model.scheduler).__name__
                if "EDM" in scheduler_type:
                    logger.info(f"Replacing {scheduler_type} with EulerDiscreteScheduler for compatibility")
                    image_model.scheduler = EulerDiscreteScheduler.from_config(
                        image_model.scheduler.config,
                        timestep_spacing="trailing",
                    )
                
            except Exception as e:
                logger.warning(f"SDXL pipeline failed, trying AutoPipeline: {e}")
                safe_progress(0.4, desc="🔄 Trying alternative loader...")
                image_model = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=cache_dir,
                )
        else:
            safe_progress(0.3, desc="📥 Loading model weights...")
            image_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
        
        safe_progress(0.75, desc="🚀 Moving to GPU...")
        image_model = image_model.to("cuda")
        
        safe_progress(0.85, desc="🔧 Applying optimizations...")
        # Disable safety checker for unrestricted generation
        if hasattr(image_model, 'safety_checker'):
            image_model.safety_checker = None
        
        # Enable memory optimizations
        image_model.enable_attention_slicing()
        
        safe_progress(0.95, desc="✨ Finalizing...")
        # Track loaded model
        loaded_image_models[model_key] = image_model
        current_image_model_key = model_key
        image_model_loaded = True
        logger.info(f"Image model loaded: {model_key}")
        
        safe_progress(1.0, desc="Done!")
        # Return full model info with loaded status
        return get_image_model_info(model_name)
        
    except ImportError:
        return "### ❌ Please install diffusers: `pip install diffusers transformers accelerate`"
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        return f"### ❌ Error loading model: {str(e)}"


def generate_image(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, lora_name, lora_strength, progress=gr.Progress()):
    """Generate an image from a text prompt."""
    global image_model, image_gen_cancel_event, image_gen_in_progress
    
    if image_model is None:
        return None, "**❌ No image model loaded**\n\nPlease select and load an image model first."
    
    if not prompt.strip():
        return None, "**❌ Missing prompt**\n\nPlease enter a prompt to generate an image."
    
    # Check for CLIP token limit using actual tokenizer if available
    tokenizer = getattr(image_model, 'tokenizer', None)
    prompt_tokens = count_clip_tokens(prompt, tokenizer)
    negative_tokens = count_clip_tokens(negative_prompt, tokenizer) if negative_prompt else 0
    token_warning = ""
    if prompt_tokens > CLIP_MAX_TOKENS:
        token_warning = f"### ⚠️ WARNING: Prompt ({prompt_tokens} tokens) exceeds {CLIP_MAX_TOKENS} limit - end of prompt will be ignored!\n\n"
    if negative_tokens > CLIP_MAX_TOKENS:
        token_warning += f"### ⚠️ Negative prompt ({negative_tokens} tokens) also truncated!\n\n"
    
    try:
        progress(0.1, desc="Preparing generation...")
        
        # Load LoRA if specified
        lora_loaded = False
        if lora_name and lora_name != "None":
            try:
                lora_path = f"data/lora_models/{lora_name}"
                if os.path.exists(lora_path):
                    progress(0.15, desc=f"Loading LoRA: {lora_name}...")
                    print(f"[LoRA] Loading from: {lora_path}")
                    image_model.load_lora_weights(lora_path)
                    image_model.fuse_lora(lora_scale=lora_strength)
                    lora_loaded = True
                    print(f"[LoRA] ✓ Loaded '{lora_name}' with strength {lora_strength}")
                    logger.info(f"Loaded LoRA: {lora_name} with strength {lora_strength}")
                else:
                    print(f"[LoRA] ✗ Path not found: {lora_path}")
            except Exception as e:
                print(f"[LoRA] ✗ Failed to load: {e}")
                logger.warning(f"Failed to load LoRA {lora_name}: {e}")
                # Continue without LoRA
        
        # Set seed for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        gen_start = time.time()
        
        progress(0.2, desc="Generating image...")

        # Setup cancel flag and callback
        image_gen_cancel_event = False
        image_gen_in_progress = True

        def _generation_callback(pipe, step: int, timestep: int, callback_kwargs):
            """Callback for step-by-step progress updates during generation."""
            # Check for cancellation
            if image_gen_cancel_event:
                raise Exception("Generation cancelled by user")
            
            # Calculate progress: reserve 0.2-0.90 range for generation steps
            # (0-0.2 is prep, 0.90-1.0 is VAE decode + saving)
            step_progress = 0.2 + (step / num_steps) * 0.70
            progress(step_progress, desc=f"Generating... step {step}/{num_steps}")
            
            return callback_kwargs

        # Generate image (pass callback to allow cancellation and progress tracking)
        progress(0.2, desc="Starting generation...")
        result = image_model(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=_generation_callback,
        )
        
        # VAE decoding happens inside the pipeline call, so we're already past it
        progress(0.92, desc="Processing image...")
        
        gen_time = time.time() - gen_start
        
        # Unload LoRA after generation to keep model clean for next generation
        if lora_loaded:
            progress(0.94, desc="Unloading LoRA...")
            try:
                image_model.unfuse_lora()
                image_model.unload_lora_weights()
            except Exception as e:
                logger.warning(f"Failed to unload LoRA: {e}")
        
        progress(0.94, desc="Processing image...")
        
        image = result.images[0]
        
        progress(0.96, desc="Saving image...")
        
        # Save to file with metadata
        os.makedirs("data/generated_images", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data/generated_images/img_{timestamp}.png"
        
        # Add generation parameters as PNG metadata
        from PIL import PngImagePlugin
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("negative_prompt", negative_prompt if negative_prompt else "")
        metadata.add_text("steps", str(num_steps))
        metadata.add_text("guidance_scale", str(guidance_scale))
        metadata.add_text("width", str(width))
        metadata.add_text("height", str(height))
        metadata.add_text("seed", str(seed))
        metadata.add_text("model", current_image_model_key if current_image_model_key else "unknown")
        metadata.add_text("lora", lora_name if lora_name and lora_name != "None" else "")
        metadata.add_text("lora_strength", str(lora_strength) if lora_name and lora_name != "None" else "")
        metadata.add_text("generated_at", timestamp)
        
        image.save(filename, pnginfo=metadata)
        
        progress(1.0, desc="Complete!")
        
        status = f"✅ Generated in **{gen_time:.1f}s** | Saved to `{filename}`"
        if lora_name and lora_name != "None":
            status += f" | LoRA: {lora_name} ({lora_strength})"
        if seed != -1:
            status += f" | Seed: {seed}"
        if token_warning:
            status = token_warning + "\n" + status
        image_gen_in_progress = False
        return image, status
        
    except Exception as e:
        image_gen_in_progress = False
        # Unload LoRA on error too
        if lora_name and lora_name != "None":
            try:
                image_model.unfuse_lora()
                image_model.unload_lora_weights()
            except:
                pass
        # If cancelled by user, return a friendly message
        if str(e).lower().find('cancel') != -1:
            logger.info("Image generation cancelled by user")
            return None, "🛑 Generation cancelled"
        logger.error(f"Error generating image: {e}")
        return None, f"❌ Error: {str(e)}"


def unload_image_model(model_name=None):
    """Unload the image model to free VRAM."""
    global image_model, image_model_loaded, loaded_image_models

    # Extract model key from selection (remove status icon)
    model_key = extract_model_key(model_name) if model_name else None
    
    if loaded_image_models or image_model is not None:
        unload_all_image_models()
        if model_key:
            return f"### ✅ Unloaded {model_key} - VRAM freed"
        return "### ✅ Image model unloaded - VRAM freed"

    return "### ℹ️ No image model was loaded"


def get_image_status():
    """Get image generation status."""
    global image_model_loaded
    
    lines = []
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"🎮 **VRAM:** {vram_used:.2f} / {vram_total:.1f} GB")
    
    if image_model_loaded:
        lines.append("✅ **Active image model:** Loaded")
    if loaded_image_models:
        lines.append(f"\n📦 **Cached models ({len(loaded_image_models)}):**")
        for name in loaded_image_models.keys():
            lines.append(f"  • {name}")
    else:
        lines.append("\n📦 **No cached image models**")
    # show progress flag
    if image_gen_in_progress:
        lines.append("\n⏳ Image generation: In progress")
    else:
        lines.append("\n✅ Image generation: Idle")

    return "\n".join(lines)


# Helper functions for generated images gallery
def get_generated_image_choices():
    """Return a list of generated images (filenames) sorted by mtime desc."""
    folder = os.path.abspath("data/generated_images")
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # return full paths so Gradio can display them
    return files


def get_gallery_images():
    """Return list of image paths for gallery display."""
    return get_generated_image_choices()


def get_gallery_checkbox_choices():
    """Return list of filenames for checkbox selection."""
    images = get_generated_image_choices()
    return [os.path.basename(p) for p in images]


def get_path_from_filename(filename):
    """Get full path from filename."""
    folder = os.path.abspath("data/generated_images")
    return os.path.join(folder, filename)


def select_all_images():
    """Select all images in the gallery."""
    choices = get_gallery_checkbox_choices()
    return gr.update(choices=choices, value=choices), f"✅ Selected {len(choices)} images"


def clear_image_selection():
    """Clear all selected images."""
    return gr.update(choices=get_gallery_checkbox_choices(), value=[]), "☐ Selection cleared"


def delete_selected_images(selected_filenames):
    """Delete multiple selected images."""
    if not selected_filenames:
        return get_gallery_images(), gr.update(choices=get_gallery_checkbox_choices(), value=[]), "⚠️ No images selected"
    
    deleted = 0
    errors = []
    for filename in selected_filenames:
        path = get_path_from_filename(filename)
        try:
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
        except Exception as e:
            errors.append(f"{filename}: {e}")
    
    # Refresh choices after deletion
    new_choices = get_gallery_checkbox_choices()
    status = f"🗑️ Deleted {deleted} image(s)"
    if errors:
        status += f" | Errors: {len(errors)}"
    
    return get_gallery_images(), gr.update(choices=new_choices, value=[]), status


def download_selected_images(selected_filenames):
    """Prepare selected images for download as a zip file."""
    import zipfile
    import tempfile
    
    if not selected_filenames:
        return None, "⚠️ No images selected"
    
    if len(selected_filenames) == 1:
        # Single file - return directly
        path = get_path_from_filename(selected_filenames[0])
        if os.path.exists(path):
            return path, f"⬇️ Downloading {selected_filenames[0]}"
        return None, "❌ File not found"
    
    # Multiple files - create zip
    try:
        # Create a zip file in temp directory
        zip_path = os.path.join(tempfile.gettempdir(), "kvgenius_images.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in selected_filenames:
                path = get_path_from_filename(filename)
                if os.path.exists(path):
                    zf.write(path, filename)
        
        return zip_path, f"⬇️ Downloading {len(selected_filenames)} images as ZIP"
    except Exception as e:
        return None, f"❌ Error creating ZIP: {e}"


def refresh_gallery():
    """Refresh both gallery and checkbox list."""
    images = get_gallery_images()
    choices = get_gallery_checkbox_choices()
    return images, gr.update(choices=choices, value=[]), f"🔄 Refreshed - {len(images)} images"


# Track currently selected image in gallery for metadata display
selected_gallery_image_path = None


def on_gallery_select(evt: gr.SelectData):
    """Handle gallery image selection - show metadata."""
    global selected_gallery_image_path
    
    if evt.value is None:
        selected_gallery_image_path = None
        return "*Click an image to see its generation settings*", gr.update(visible=False)
    
    # Get the image path from the selection
    # evt.value is a dict with 'image' containing the path
    if isinstance(evt.value, dict):
        img_path = evt.value.get('image', {}).get('path', '')
    elif isinstance(evt.value, str):
        img_path = evt.value
    else:
        img_path = str(evt.value)
    
    selected_gallery_image_path = img_path
    
    # Read and format metadata
    metadata = read_image_metadata(img_path)
    if metadata:
        formatted = format_image_metadata(metadata)
        return formatted, gr.update(visible=True)
    else:
        return f"*No metadata found for this image*\n\n`{os.path.basename(img_path)}`", gr.update(visible=False)


def load_prompt_from_gallery_image():
    """Load the prompt and settings from the selected gallery image into the generator."""
    global selected_gallery_image_path
    
    if not selected_gallery_image_path:
        return [gr.update()] * 6 + ["⚠️ No image selected"]
    
    metadata = read_image_metadata(selected_gallery_image_path)
    if not metadata:
        return [gr.update()] * 6 + ["⚠️ No metadata found in image"]
    
    # Return updates for: prompt, negative, steps, guidance, width, height, status
    return (
        gr.update(value=metadata.get('prompt', '')),
        gr.update(value=metadata.get('negative_prompt', '')),
        gr.update(value=int(metadata.get('steps', 25))),
        gr.update(value=float(metadata.get('guidance_scale', 7.5))),
        gr.update(value=int(metadata.get('width', 512))),
        gr.update(value=int(metadata.get('height', 512))),
        f"✅ Loaded prompt from `{os.path.basename(selected_gallery_image_path)}`"
    )


def delete_gallery_image(path):
    """Delete an image from the gallery."""
    if not path:
        return get_gallery_images(), "", "⚠️ No image selected"
    try:
        if os.path.exists(path):
            filename = os.path.basename(path)
            os.remove(path)
            return get_gallery_images(), "", f"🗑️ Deleted {filename}"
        return get_gallery_images(), "", "ℹ️ File not found"
    except Exception as e:
        return get_gallery_images(), "", f"❌ Error deleting: {e}"


# ==================== Prompt Library Functions ====================

def get_prompt_library_choices():
    """Get list of saved prompts for display."""
    prompts = db.get_all_prompts()
    return [f"📝 {p['name']}" for p in prompts]


def save_current_prompt(name, prompt, negative, steps, guidance, width, height):
    """Save current generation settings to prompt library."""
    if not name or not name.strip():
        return gr.update(), "❌ Please enter a name for the prompt"
    
    if not prompt or not prompt.strip():
        return gr.update(), "❌ Prompt cannot be empty"
    
    try:
        db.save_prompt(
            name=name.strip(),
            prompt=prompt,
            negative_prompt=negative,
            steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            model=current_image_model_key
        )
        # Refresh the list
        choices = get_prompt_library_choices()
        return gr.update(choices=choices), f"✅ Saved prompt '{name}'"
    except Exception as e:
        return gr.update(), f"❌ Error saving: {e}"


def load_prompt_from_library(selection):
    """Load a saved prompt into the generator."""
    if not selection:
        return [gr.update()] * 7 + ["⚠️ No prompt selected"]
    
    # Extract name from selection (remove emoji prefix)
    name = selection.replace("📝 ", "").strip()
    prompt_data = db.get_prompt_by_name(name)
    
    if not prompt_data:
        return [gr.update()] * 7 + ["❌ Prompt not found"]
    
    return (
        gr.update(value=prompt_data.get('prompt', '')),
        gr.update(value=prompt_data.get('negative_prompt', '')),
        gr.update(value=int(prompt_data.get('steps', 25))),
        gr.update(value=float(prompt_data.get('guidance_scale', 7.5))),
        gr.update(value=int(prompt_data.get('width', 512))),
        gr.update(value=int(prompt_data.get('height', 768))),
        gr.update(value=""),  # Clear save name
        f"✅ Loaded prompt '{name}'"
    )


def delete_prompt_from_library(selection):
    """Delete a prompt from the library."""
    if not selection:
        return gr.update(), "⚠️ No prompt selected"
    
    name = selection.replace("📝 ", "").strip()
    prompt_data = db.get_prompt_by_name(name)
    
    if prompt_data:
        db.delete_prompt(prompt_data['id'])
        choices = get_prompt_library_choices()
        return gr.update(choices=choices, value=None), f"🗑️ Deleted prompt '{name}'"
    
    return gr.update(), "❌ Prompt not found"


def format_prompt_preview(selection):
    """Show preview of selected prompt."""
    if not selection:
        return "*Select a prompt to see details*"
    
    name = selection.replace("📝 ", "").strip()
    prompt_data = db.get_prompt_by_name(name)
    
    if not prompt_data:
        return "*Prompt not found*"
    
    lines = []
    lines.append(f"### 📝 {prompt_data['name']}")
    lines.append("")
    lines.append(f"**Prompt:**\n{prompt_data['prompt']}")
    
    if prompt_data.get('negative_prompt'):
        lines.append(f"\n**Negative:**\n{prompt_data['negative_prompt']}")
    
    settings = []
    settings.append(f"Steps: {prompt_data.get('steps', 25)}")
    settings.append(f"CFG: {prompt_data.get('guidance_scale', 7.5)}")
    settings.append(f"Size: {prompt_data.get('width', 512)}x{prompt_data.get('height', 768)}")
    if prompt_data.get('model'):
        settings.append(f"Model: {prompt_data['model']}")
    
    lines.append(f"\n**Settings:** {' | '.join(settings)}")
    
    return "\n".join(lines)


# ==================== LoRA Training UI Functions ====================

def get_lora_dataset_choices():
    """Get list of available datasets for dropdown."""
    dm = get_dataset_manager()
    datasets = dm.list_datasets()
    return datasets if datasets else []


def get_lora_choices():
    """Get list of trained LoRAs for dropdown."""
    lm = get_lora_manager()
    loras = lm.list_loras()
    return [l.name for l in loras] if loras else []


def create_new_dataset(name):
    """Create a new training dataset."""
    if not name or not name.strip():
        return "❌ Please enter a dataset name", gr.update()
    
    name = name.strip().replace(" ", "_")
    dm = get_dataset_manager()
    
    try:
        dm.create_dataset(name)
        return f"✅ Created dataset: {name}", gr.update(choices=get_lora_dataset_choices(), value=name)
    except Exception as e:
        return f"❌ Error: {e}", gr.update()


def delete_dataset(name):
    """Delete a dataset."""
    if not name:
        return "❌ Select a dataset first", gr.update(), []
    
    dm = get_dataset_manager()
    dm.delete_dataset(name)
    return f"✅ Deleted dataset: {name}", gr.update(choices=get_lora_dataset_choices(), value=None), []


def upload_training_images(files, dataset_name, crop_size, crop_mode, auto_caption):
    """Upload and process images for training dataset."""
    if not dataset_name:
        return "❌ Please select or create a dataset first", []
    
    if not files:
        return "❌ No files selected", []
    
    dm = get_dataset_manager()
    uploaded = 0
    
    from PIL import Image
    
    for file in files:
        try:
            img = Image.open(file.name).convert("RGB")
            
            # Process image with selected crop mode
            img = dm.crop_image(img, size=crop_size, crop_mode=crop_mode)
            
            filename = os.path.basename(file.name)
            
            # Auto-caption if enabled
            caption = ""
            if auto_caption:
                caption = auto_caption_image(img)
            
            dm.add_image(dataset_name, img, filename, caption)
            uploaded += 1
            
        except Exception as e:
            logger.error(f"Failed to upload {file.name}: {e}")
    
    # Return updated gallery
    images = dm.get_images(dataset_name)
    gallery_items = [(img["path"], img.get("caption", "")) for img in images]
    
    return f"✅ Uploaded {uploaded} images to {dataset_name}", gallery_items


def load_dataset_images(dataset_name):
    """Load images from a dataset for display."""
    if not dataset_name:
        return []
    
    dm = get_dataset_manager()
    images = dm.get_images(dataset_name)
    return [(img["path"], img.get("caption", "")[:50] + "..." if len(img.get("caption", "")) > 50 else img.get("caption", "")) for img in images]


def get_image_caption(dataset_name, evt: gr.SelectData):
    """Get caption for selected image in gallery."""
    if not dataset_name:
        return "", ""
    
    dm = get_dataset_manager()
    images = dm.get_images(dataset_name)
    
    if evt.index < len(images):
        img = images[evt.index]
        return img.get("caption", ""), img.get("filename", "")
    return "", ""


def update_image_caption(dataset_name, filename, new_caption):
    """Update caption for an image."""
    if not dataset_name or not filename:
        return "❌ Select an image first"
    
    dm = get_dataset_manager()
    if dm.update_caption(dataset_name, filename, new_caption):
        return f"✅ Caption updated for {filename}"
    return "❌ Failed to update caption"


def delete_dataset_image(dataset_name, filename):
    """Delete an image from dataset."""
    if not dataset_name or not filename:
        return "❌ Select an image first", []
    
    dm = get_dataset_manager()
    dm.delete_image(dataset_name, filename)
    
    # Return updated gallery
    images = dm.get_images(dataset_name)
    gallery_items = [(img["path"], img.get("caption", "")[:30]) for img in images]
    return f"✅ Deleted {filename}", gallery_items


def auto_caption_image(image):
    """Generate caption using BLIP model (lazy loaded)."""
    global blip_model, blip_processor
    
    try:
        if 'blip_model' not in globals() or blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Get model config from utility presets
            captioner_config = UTILITY_MODELS.get('image_captioner', {})
            model_id = captioner_config.get('id', 'Salesforce/blip-image-captioning-base')
            device = captioner_config.get('device', 'cpu')
            
            logger.info(f"Loading BLIP captioning model: {model_id} on {device}...")
            blip_processor = BlipProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir
            )
            blip_model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=cache_dir
            ).to(device)
            blip_model.eval()
        
        inputs = blip_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50)
        
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        logger.error(f"Auto-captioning failed: {e}")
        return ""


def suggest_captions_for_dataset(dataset_name):
    """Generate suggested captions for all images in dataset."""
    if not dataset_name:
        return "❌ Select a dataset first", []
    
    dm = get_dataset_manager()
    images = dm.get_images(dataset_name)
    
    if not images:
        return "❌ No images in dataset", []
    
    from PIL import Image
    updated = 0
    
    for img_data in images:
        if not img_data.get("caption"):  # Only caption images without captions
            try:
                img = Image.open(img_data["path"]).convert("RGB")
                caption = auto_caption_image(img)
                if caption:
                    dm.update_caption(dataset_name, img_data["filename"], caption)
                    updated += 1
            except Exception as e:
                logger.error(f"Failed to caption {img_data['filename']}: {e}")
    
    # Return updated gallery
    images = dm.get_images(dataset_name)
    gallery_items = [(img["path"], img.get("caption", "")[:30]) for img in images]
    return f"✅ Generated captions for {updated} images", gallery_items


def get_base_model_choices():
    """Get available base models for LoRA training."""
    # Return models that are compatible with LoRA training
    choices = []
    for key, model in AVAILABLE_IMAGE_MODELS.items():
        # Skip SDXL for now (needs different training approach)
        if "sdxl" not in key.lower():
            choices.append(key)
    return choices


def on_resume_lora_selected(lora_name):
    """Auto-fill fields when selecting a LoRA to continue training."""
    if not lora_name or lora_name == "(New LoRA)":
        return gr.update(), gr.update(), gr.update()
    
    lm = get_lora_manager()
    info = lm.get_lora(lora_name)
    
    if not info:
        return gr.update(), gr.update(), gr.update()
    
    # Auto-fill the name, trigger word, and description
    return (
        gr.update(value=lora_name),  # output_name
        gr.update(value=info.trigger_word),  # trigger_word
        gr.update(value=info.description)  # description
    )


def start_lora_training(dataset_name, output_name, resume_from, base_model, trigger_word, description,
                        num_epochs, learning_rate, lora_rank, lora_alpha, resolution, max_steps):
    """Start LoRA training in background."""
    trainer = get_trainer()
    
    if trainer.is_training():
        return "❌ Training already in progress", ""
    
    if not dataset_name:
        return "❌ Select a dataset first", ""
    
    if not output_name or not output_name.strip():
        return "❌ Enter an output name for the LoRA", ""
    
    if not trigger_word or not trigger_word.strip():
        return "❌ Enter a trigger word for the LoRA", ""
    
    dm = get_dataset_manager()
    images = dm.get_images(dataset_name)
    
    if len(images) < 3:
        return f"❌ Need at least 3 images (found {len(images)})", ""
    
    # Get full model path from the "id" field
    model_info = AVAILABLE_IMAGE_MODELS.get(base_model, {})
    model_path = model_info.get("id", base_model)  # Use "id" field for HuggingFace path
    
    # Handle resume from existing LoRA
    resume_path = None
    if resume_from and resume_from != "(New LoRA)":
        resume_path = f"data/lora_models/{resume_from}"
        if os.path.exists(resume_path):
            print(f"[Training] Continuing from existing LoRA: {resume_path}")
        else:
            resume_path = None
    
    config = TrainingConfig(
        dataset_name=dataset_name,
        output_name=output_name.strip().replace(" ", "_"),
        base_model=model_path,
        trigger_word=trigger_word.strip(),
        description=description,
        resume_from=resume_path,
        num_train_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        resolution=int(resolution),
        max_train_steps=int(max_steps) if max_steps > 0 else None
    )
    
    success = trainer.start_training(config)
    
    if success:
        status_msg = "🚀 Training started!"
        if resume_path:
            status_msg += f" (continuing from {resume_from})"
        return status_msg + " Check progress below.", f"Training {output_name}..."
    else:
        return "❌ Failed to start training", ""


def get_training_progress():
    """Get current training progress for UI update."""
    trainer = get_trainer()
    state = trainer.get_state()
    
    if not state.is_training and not state.completed and not state.error:
        return "Idle - Ready to train", 0, ""
    
    progress = 0
    if state.total_steps > 0:
        progress = (state.current_step / state.total_steps) * 100
    
    details = f"Epoch: {state.current_epoch}/{state.total_epochs} | Step: {state.current_step}/{state.total_steps} | Loss: {state.loss:.4f}"
    
    return state.status_message, progress, details


def stop_training():
    """Request training to stop."""
    trainer = get_trainer()
    trainer.request_stop()
    return "⏹ Stopping training...", 0, ""


def get_lora_info(lora_name):
    """Get detailed info for a LoRA."""
    if not lora_name:
        return "*Select a LoRA to see details*"
    
    lm = get_lora_manager()
    info = lm.get_lora(lora_name)
    
    if not info:
        return "*LoRA not found*"
    
    lines = [
        f"### {info.name}",
        f"**Trigger Word:** `{info.trigger_word}`",
        f"**Base Model:** {info.base_model}",
        f"**Training Steps:** {info.training_steps}",
        f"**Rank:** {info.rank} | **Alpha:** {info.alpha}",
        f"**Size:** {info.file_size_mb:.1f} MB",
        f"**Created:** {info.created_at[:10] if info.created_at else 'Unknown'}",
    ]
    
    if info.description:
        lines.append(f"\n*{info.description}*")
    
    return "\n\n".join(lines)


def delete_lora(lora_name):
    """Delete a trained LoRA."""
    if not lora_name:
        return "❌ Select a LoRA first", gr.update()
    
    lm = get_lora_manager()
    lm.delete_lora(lora_name)
    
    return f"✅ Deleted LoRA: {lora_name}", gr.update(choices=get_lora_choices(), value=None)


def import_lora(file, lora_name, trigger_word):
    """Import a pre-built LoRA .safetensors file."""
    import shutil
    from pathlib import Path
    
    if file is None:
        return "❌ Please select a .safetensors file", gr.update()
    
    if not lora_name or not lora_name.strip():
        return "❌ Please enter a name for the LoRA", gr.update()
    
    lora_name = lora_name.strip().replace(" ", "_")
    
    # Validate file type
    if not file.name.lower().endswith('.safetensors'):
        return "❌ File must be a .safetensors file", gr.update()
    
    try:
        # Create LoRA directory
        lora_dir = Path("data/lora_models") / lora_name
        lora_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        dest_file = lora_dir / "adapter_model.safetensors"
        shutil.copy2(file.name, dest_file)
        
        # Try to detect rank from the file
        rank = 16  # default
        try:
            from safetensors import safe_open
            with safe_open(str(dest_file), framework="pt") as f:
                for key in f.keys():
                    if "lora_down" in key.lower() or "lora_a" in key.lower():
                        tensor = f.get_tensor(key)
                        rank = min(tensor.shape)
                        break
        except Exception as e:
            logger.warning(f"Could not detect LoRA rank: {e}")
        
        # Create adapter_config.json
        adapter_config = {
            "base_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "inference_mode": True,
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": rank * 2,
            "lora_dropout": 0.0,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            "bias": "none"
        }
        
        with open(lora_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        # Create metadata
        if not trigger_word or not trigger_word.strip():
            trigger_word = lora_name.lower()
        
        metadata = {
            "name": lora_name,
            "trigger_word": trigger_word.strip(),
            "source": os.path.basename(file.name),
            "installed_from": "imported",
            "rank": rank,
            "description": f"Imported from {os.path.basename(file.name)}"
        }
        
        with open(lora_dir / "lora_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Imported LoRA: {lora_name} (rank={rank}, trigger={trigger_word})")
        
        return f"✅ Imported '{lora_name}'! Trigger word: **{trigger_word}**", gr.update(choices=get_lora_choices(), value=lora_name)
        
    except Exception as e:
        logger.error(f"Error importing LoRA: {e}")
        return f"❌ Error: {str(e)}", gr.update()


# ==================== Cards Against Humanity Generator Functions ====================

def get_cah_generator():
    """Get or create the CAH generator instance."""
    global cah_generator
    if cah_generator is None:
        cah_generator = CAHGenerator()
    return cah_generator


def get_cah_style_choices():
    """Get style choices for dropdown."""
    return [
        ("🎭 Classic CAH - Irreverent & crude humor", "classic"),
        ("🌀 Absurd - Surreal & nonsensical", "absurd"),
        ("🖤 Dark - Gallows humor & morbid", "dark"),
        ("✨ Wholesome - Family-friendly", "wholesome"),
        ("🎮 Nerdy - Gaming, tech, pop culture", "nerdy"),
        ("✏️ Custom - Define your own style", "custom"),
    ]


def get_cah_type_choices():
    """Get card type choices for dropdown."""
    return [
        ("⬛ Black Cards (Prompts with blanks)", "black"),
        ("⬜ White Cards (Answer cards)", "white"),
        ("🃏 Both Types", "both"),
    ]


def generate_cah_cards(topic, card_type, quantity, style, custom_style, progress=gr.Progress()):
    """Generate CAH-style cards using the loaded chat model."""
    global active_models
    import time
    
    if not topic or not topic.strip():
        return [], "❌ Please enter a topic or theme"
    
    if not active_models:
        return [], "❌ No text model loaded. Please load a model first."
    
    # Get the first loaded model
    model_key = list(active_models.keys())[0]
    model, tokenizer, chatbot = active_models[model_key]
    
    # Diagnostic: Check model device
    model_device = next(model.parameters()).device
    logger.info(f"[CARD-GEN] Model '{model_key}' is on device: {model_device}")
    if str(model_device) == "cpu":
        logger.warning("[CARD-GEN] ⚠️ MODEL IS ON CPU - THIS WILL BE VERY SLOW!")
        return [], f"**⚠️ Model is on CPU!**\n\n{model_key} loaded on CPU instead of GPU. This would take 10+ minutes. Please unload and reload the model."
    
    progress(0.1, desc="Preparing generation...")
    
    # Build the prompts
    cah = get_cah_generator()
    style_enum = CardStyle(style)
    type_enum = CardType(card_type)
    
    system_prompt = cah.get_system_prompt(style_enum, custom_style if style == "custom" else "")
    user_prompt = get_generation_prompt(topic, type_enum, quantity, style_enum, custom_style if style == "custom" else "")
    
    progress(0.2, desc=f"Generating {quantity} cards with {model_key}...")
    
    try:
        # Format prompt for the model
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        # Generate
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info(f"[CARD-GEN] Input tokens: {inputs['input_ids'].shape[1]}")
        logger.info(f"[CARD-GEN] Inputs on device: {inputs['input_ids'].device}")
        
        progress(0.4, desc="Generating response...")
        
        gen_start = time.time()
        
        # Scale max tokens based on quantity requested (~40 tokens per card is plenty)
        max_tokens = min(100 + (quantity * 50), 800)
        logger.info(f"[CARD-GEN] Starting generation (max {max_tokens} tokens for {quantity} cards)...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        gen_elapsed = time.time() - gen_start
        output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = output_tokens / gen_elapsed if gen_elapsed > 0 else 0
        logger.info(f"[CARD-GEN] Generation completed in {gen_elapsed:.2f}s ({output_tokens} tokens, {tokens_per_sec:.1f} tok/s)")
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Log first 500 chars of response for debugging
        logger.info(f"[CARD-GEN] Raw response (first 500 chars):\n{response[:500]}")
        
        progress(0.8, desc="Parsing cards...")
        
        # Parse the response into cards - pass force_type if user selected specific type
        force_type = type_enum if type_enum != CardType.BOTH else None
        cards = cah.parse_generated_cards(response, topic, style_enum, force_type)
        
        if cards:
            # Save the cards
            cah.add_cards(cards)
            progress(1.0, desc="Done!")
            
            # Format for display
            display_cards = []
            for card in cards:
                card_emoji = "⬛" if card.card_type == "black" else "⬜"
                display_cards.append(f"{card_emoji} {card.text}")
            
            return display_cards, f"✅ Generated {len(cards)} cards! ({len([c for c in cards if c.card_type == 'black'])} black, {len([c for c in cards if c.card_type == 'white'])} white)"
        else:
            return [], "⚠️ Could not parse cards from response. Try again or adjust the topic."
            
    except Exception as e:
        logger.error(f"Error generating cards: {e}")
        import traceback
        traceback.print_exc()
        return [], f"❌ Error: {str(e)}"


def fetch_article_from_url(url):
    """Fetch and extract article text from a URL."""
    if not url or not url.strip():
        return "", "❌ Please enter a URL"
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script, style, nav, header, footer elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'iframe']):
            element.decompose()
        
        # Try to find the main article content
        article_text = ""
        
        # Look for common article containers
        article_selectors = [
            'article',
            '[role="article"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.story-body',
            '.article-body',
            'main',
            '.content'
        ]
        
        for selector in article_selectors:
            content = soup.select_one(selector)
            if content:
                # Get all paragraph text
                paragraphs = content.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                if paragraphs:
                    article_text = '\n\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    if len(article_text) > 200:  # Found substantial content
                        break
        
        # Fallback: get all paragraphs
        if len(article_text) < 200:
            paragraphs = soup.find_all('p')
            article_text = '\n\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        if len(article_text) < 100:
            return "", "⚠️ Could not extract enough text from this URL. Try pasting the article text manually."
        
        # Get title if available
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        if title:
            article_text = f"# {title}\n\n{article_text}"
        
        return article_text, f"✅ Fetched article ({len(article_text)} characters)"
        
    except requests.exceptions.Timeout:
        return "", "❌ Request timed out. Try again or paste the article text manually."
    except requests.exceptions.RequestException as e:
        return "", f"❌ Failed to fetch URL: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching article: {e}")
        return "", f"❌ Error: {str(e)}"


def extract_cah_from_article(article_text, card_type, quantity, progress=gr.Progress()):
    """Extract CAH-style cards from an article using the loaded chat model."""
    global active_models
    
    if not article_text or not article_text.strip():
        return [], "❌ Please paste an article to extract jokes from"
    
    if len(article_text.strip()) < 100:
        return [], "❌ Article seems too short. Please paste a full article."
    
    if not active_models:
        return [], "❌ No text model loaded. Please load a model first."
    
    # Get the first loaded model
    model_key = list(active_models.keys())[0]
    model, tokenizer, chatbot = active_models[model_key]
    
    progress(0.1, desc="Analyzing article...")
    
    # Build the prompts
    cah = get_cah_generator()
    type_enum = CardType(card_type)
    
    system_prompt = ARTICLE_EXTRACTION_PROMPT
    user_prompt = get_article_extraction_prompt(article_text, type_enum, quantity)
    
    progress(0.2, desc=f"Extracting jokes with {model_key}...")
    
    try:
        # Format prompt for the model
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        # Generate
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        progress(0.4, desc="Generating response...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.7,  # Slightly lower for more faithful extraction
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        progress(0.8, desc="Parsing cards...")
        
        # Parse the response into cards - pass force_type if user selected specific type
        force_type = type_enum if type_enum != CardType.BOTH else None
        cards = cah.parse_generated_cards(response, "Article extraction", CardStyle.ABSURD, force_type)
        
        if cards:
            # Save the cards
            cah.add_cards(cards)
            progress(1.0, desc="Done!")
            
            # Format for display
            display_cards = []
            for card in cards:
                card_emoji = "⬛" if card.card_type == "black" else "⬜"
                display_cards.append(f"{card_emoji} {card.text}")
            
            return display_cards, f"✅ Extracted {len(cards)} cards from article! ({len([c for c in cards if c.card_type == 'black'])} black, {len([c for c in cards if c.card_type == 'white'])} white)"
        else:
            return [], "⚠️ Could not extract cards from article. Try a different article or adjust quantity."
            
    except Exception as e:
        logger.error(f"Error extracting cards: {e}")
        import traceback
        traceback.print_exc()
        return [], f"❌ Error: {str(e)}"


def get_saved_cards_display():
    """Get saved cards for display in gallery with IDs for deletion."""
    cah = get_cah_generator()
    cards = cah.get_cards()
    
    display = []
    for card in reversed(cards):  # Most recent first
        card_emoji = "⬛" if card.card_type == "black" else "⬜"
        star = "⭐ " if card.favorited else ""
        display.append(f"{star}{card_emoji} {card.text}")
    
    return display


def get_saved_cards_with_ids():
    """Get saved cards with IDs for the deletion dropdown."""
    cah = get_cah_generator()
    cards = cah.get_cards()
    
    choices = []
    for card in reversed(cards):  # Most recent first
        card_emoji = "⬛" if card.card_type == "black" else "⬜"
        # Truncate long cards for the dropdown
        text_preview = card.text[:50] + "..." if len(card.text) > 50 else card.text
        choices.append(f"{card_emoji} {text_preview} [{card.id}]")
    
    return choices


def delete_selected_cards(selected_cards):
    """Delete selected cards from the library."""
    if not selected_cards:
        return get_saved_cards_display(), "⚠️ No cards selected"
    
    cah = get_cah_generator()
    deleted = 0
    
    for selection in selected_cards:
        # Extract card ID from the selection string - it's in brackets at the end
        if '[' in selection and ']' in selection:
            card_id = selection[selection.rfind('[')+1:selection.rfind(']')]
            if cah.delete_card(card_id):
                deleted += 1
    
    return get_saved_cards_display(), f"✅ Deleted {deleted} card(s)"


def delete_last_n_cards(n):
    """Delete the last N generated cards."""
    cah = get_cah_generator()
    cards = cah.get_cards()
    
    if not cards:
        return get_saved_cards_display(), "⚠️ No cards to delete"
    
    n = min(n, len(cards))
    deleted = 0
    
    # Get the last N card IDs (most recent are at the end of the list)
    cards_to_delete = cards[-n:]
    for card in cards_to_delete:
        if cah.delete_card(card.id):
            deleted += 1
    
    return get_saved_cards_display(), f"✅ Deleted {deleted} most recent card(s)"


def get_cah_stats():
    """Get stats about saved cards."""
    cah = get_cah_generator()
    stats = cah.get_stats()
    
    if stats['total'] == 0:
        return "### 📊 Card Statistics\n\n*No cards generated yet. Create some!*"
    
    lines = [
        "### 📊 Card Statistics",
        "",
        f"**Total Cards:** {stats['total']}",
        f"- ⬛ Black Cards: {stats['black_cards']}",
        f"- ⬜ White Cards: {stats['white_cards']}",
        f"- ⭐ Favorited: {stats['favorited']}",
        "",
        "**By Style:**"
    ]
    
    for style, count in stats['styles'].items():
        lines.append(f"- {style}: {count}")
    
    if stats['topics']:
        lines.append("")
        lines.append("**Top Topics:**")
        for topic, count in list(stats['topics'].items())[:5]:
            lines.append(f"- {topic}: {count}")
    
    return "\n".join(lines)


def export_cah_cards(format_type):
    """Export all cards in the specified format."""
    cah = get_cah_generator()
    
    if not cah.cards:
        return None, "⚠️ No cards to export"
    
    export_data = cah.export_cards(format=format_type)
    
    # Save to temp file
    import tempfile
    ext = {"json": ".json", "txt": ".txt", "csv": ".csv"}.get(format_type, ".txt")
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8')
    temp_file.write(export_data)
    temp_file.close()
    
    return temp_file.name, f"✅ Exported {len(cah.cards)} cards as {format_type.upper()}"


def clear_all_cah_cards():
    """Clear all saved cards."""
    cah = get_cah_generator()
    count = len(cah.cards)
    cah.cards = []
    cah._save_cards()
    return [], f"🗑️ Deleted {count} cards"


def refresh_cah_cards():
    """Refresh the cards display."""
    return get_saved_cards_display(), get_cah_stats()


# Initialize BLIP model variables
blip_model = None
blip_processor = None


# ==================== CPU Prompt Enhancer ====================

# Prompt enhancement templates for different styles
ENHANCEMENT_TEMPLATES = {
    "photorealistic": {
        "prefix": "",
        "suffix": ", RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "quality": ", masterpiece, best quality, highly detailed, sharp focus"
    },
    "artistic": {
        "prefix": "",
        "suffix": ", digital art, concept art, smooth, sharp focus",
        "quality": ", masterpiece, best quality, highly detailed, dramatic lighting, vibrant colors"
    },
    "portrait": {
        "prefix": "portrait of ",
        "suffix": ", professional portrait photography, studio lighting, 85mm lens, bokeh",
        "quality": ", masterpiece, detailed skin texture, sharp eyes, natural lighting"
    },
    "fantasy": {
        "prefix": "",
        "suffix": ", fantasy art, epic scene, magical atmosphere, ethereal lighting",
        "quality": ", masterpiece, best quality, highly detailed, dramatic composition"
    },
    "anime": {
        "prefix": "",
        "suffix": ", anime style, cel shading, vibrant colors",
        "quality": ", masterpiece, best quality, highly detailed, sharp lines"
    }
}


def load_prompt_enhancer():
    """Lazy load a small T5 model for prompt enhancement on CPU."""
    global prompt_enhancer_model, prompt_enhancer_tokenizer
    
    if prompt_enhancer_model is not None:
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Get model from utility config
        config = UTILITY_MODELS.get('prompt_enhancer', {})
        model_name = config.get('id', 'google/flan-t5-small')
        device = config.get('device', 'cpu')
        
        logger.info(f"Loading prompt enhancer ({model_name}) on {device}...")
        
        prompt_enhancer_tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        prompt_enhancer_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        # Keep on CPU explicitly to preserve GPU VRAM
        prompt_enhancer_model = prompt_enhancer_model.to(device)
        prompt_enhancer_model.eval()
        
        logger.info(f"Prompt enhancer loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load prompt enhancer: {e}")
        return False


def enhance_prompt_with_ai(prompt, style="photorealistic"):
    """Use AI to enhance a basic prompt into a more detailed one."""
    global prompt_enhancer_model, prompt_enhancer_tokenizer
    
    if not prompt or not prompt.strip():
        return prompt, "❌ Please enter a prompt first"
    
    # Try to load model if not loaded
    if prompt_enhancer_model is None:
        if not load_prompt_enhancer():
            # Fallback to template-based enhancement
            return enhance_prompt_with_template(prompt, style)
    
    try:
        # Style-specific context for better AI responses
        style_hints = {
            "photorealistic": "realistic photograph, natural lighting, detailed textures",
            "artistic": "digital artwork, vibrant colors, creative composition",
            "portrait": "portrait photography, facial details, studio lighting",
            "fantasy": "fantasy art, magical elements, epic atmosphere",
            "anime": "anime illustration, cel shading, expressive style"
        }
        style_hint = style_hints.get(style, style_hints["photorealistic"])
        
        # Better instruction that asks for specific additions
        instruction = f"""Add specific visual details to this image prompt. Include: setting/environment, lighting conditions, colors, textures, camera angle or composition. Style: {style_hint}

Prompt: {prompt.strip()}

Enhanced prompt:"""
        
        inputs = prompt_enhancer_tokenizer(
            instruction,
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = prompt_enhancer_model.generate(
                inputs.input_ids,
                max_length=100,
                num_beams=4,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        ai_output = prompt_enhancer_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Check if AI gave a meaningful response (not just generic text)
        generic_responses = ["masterpiece", "best quality", "highly detailed", "sharp focus"]
        is_generic = all(term in ai_output.lower() for term in generic_responses[:2]) and len(ai_output) < len(prompt) + 50
        
        if is_generic or len(ai_output) < 10:
            # AI didn't help much, use template with smart expansion
            return enhance_prompt_smart(prompt, style)
        
        # Combine original prompt idea with AI expansion
        # If AI output seems like a complete rewrite, use it; otherwise combine
        if prompt.lower().split()[0] in ai_output.lower():
            enhanced = ai_output
        else:
            enhanced = f"{prompt.strip()}, {ai_output}"
        
        # Add minimal style suffix (not the heavy quality terms)
        template = ENHANCEMENT_TEMPLATES.get(style, ENHANCEMENT_TEMPLATES["photorealistic"])
        enhanced += template["suffix"]
        
        token_count = count_clip_tokens(enhanced, None)
        return enhanced, f"✨ AI Enhanced! ({token_count}/77 tokens)"
        
    except Exception as e:
        logger.error(f"AI enhancement failed: {e}")
        # Fallback to smart template
        return enhance_prompt_smart(prompt, style)


def enhance_prompt_smart(prompt, style="photorealistic"):
    """Smart enhancement that adds contextually relevant details."""
    if not prompt or not prompt.strip():
        return prompt, "❌ Please enter a prompt first"
    
    prompt_lower = prompt.lower()
    additions = []
    
    # Context-aware additions based on prompt content
    if any(word in prompt_lower for word in ["woman", "man", "person", "girl", "boy", "portrait"]):
        additions.extend(["detailed face", "expressive eyes"])
    if any(word in prompt_lower for word in ["landscape", "mountain", "forest", "ocean", "city"]):
        additions.extend(["atmospheric perspective", "volumetric lighting"])
    if any(word in prompt_lower for word in ["night", "dark", "evening"]):
        additions.extend(["dramatic shadows", "rim lighting"])
    if any(word in prompt_lower for word in ["sunset", "sunrise", "golden"]):
        additions.extend(["golden hour", "warm tones"])
    
    # Style-specific additions
    style_additions = {
        "photorealistic": ["RAW photo", "8k uhd", "dslr", "natural lighting"],
        "artistic": ["digital painting", "artstation trending", "vibrant colors"],
        "portrait": ["studio lighting", "85mm lens", "shallow depth of field"],
        "fantasy": ["magical atmosphere", "ethereal glow", "epic composition"],
        "anime": ["anime style", "cel shading", "clean lines"]
    }
    additions.extend(style_additions.get(style, style_additions["photorealistic"])[:3])
    
    # Build enhanced prompt
    enhanced = prompt.strip()
    if additions:
        enhanced += ", " + ", ".join(additions[:5])  # Limit to avoid token overflow
    
    token_count = count_clip_tokens(enhanced, None)
    return enhanced, f"✨ Smart enhanced! ({token_count}/77 tokens)"


def enhance_prompt_with_template(prompt, style="photorealistic"):
    """Enhance prompt using predefined templates (no AI required)."""
    if not prompt or not prompt.strip():
        return prompt, "❌ Please enter a prompt first"
    
    template = ENHANCEMENT_TEMPLATES.get(style, ENHANCEMENT_TEMPLATES["photorealistic"])
    
    enhanced = prompt.strip()
    if template["prefix"] and not enhanced.lower().startswith(template["prefix"]):
        enhanced = template["prefix"] + enhanced
    # Only add suffix, skip the generic "quality" terms
    enhanced += template["suffix"]
    
    token_count = count_clip_tokens(enhanced, None)
    return enhanced, f"✨ Quick {style} style! ({token_count}/77 tokens)"


def detect_style_from_prompt(prompt):
    """Auto-detect style from prompt content."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["anime", "manga", "cel shading", "cartoon"]):
        return "anime"
    elif any(word in prompt_lower for word in ["fantasy", "magical", "dragon", "wizard", "fairy", "ethereal"]):
        return "fantasy"
    elif any(word in prompt_lower for word in ["portrait", "headshot", "face", "person closeup"]):
        return "portrait"
    elif any(word in prompt_lower for word in ["painting", "artwork", "artistic", "illustration", "concept art"]):
        return "artistic"
    else:
        return "photorealistic"


def quick_enhance_prompt(prompt):
    """Quick enhancement using templates only (fast, no model loading)."""
    style = detect_style_from_prompt(prompt)
    return enhance_prompt_with_template(prompt, style)


def ai_enhance_prompt(prompt):
    """Full AI enhancement (loads model if needed)."""
    style = detect_style_from_prompt(prompt)
    return enhance_prompt_with_ai(prompt, style)


def update_img_model_controls(model_name):
    """Return model info text, button visibility, slider updates, and generate button state based on model presets."""
    # Extract model key from selection (remove status icon)
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_IMAGE_MODELS:
        return (
            "*Select a model*", 
            gr.update(visible=True),   # load button
            gr.update(visible=False),  # unload button
            gr.update(),               # steps slider
            gr.update(),               # guidance slider
            gr.update(),               # width slider
            gr.update(),               # height slider
            gr.update(),               # negative prompt
            gr.update(interactive=False)  # generate button disabled
        )

    m = AVAILABLE_IMAGE_MODELS[model_key]
    tips = f"\n\n💡 *{m.get('tips', '')}*" if m.get('tips') else ""
    info = f"*{m['description']}*\n\n**VRAM:** {m['vram']}{tips}"
    
    # Get preset values
    step_range = m.get('step_range', [1, 50])
    guidance_range = m.get('guidance_range', [0.0, 20.0])
    
    # Build slider updates
    steps_update = gr.update(
        value=m.get('default_steps', 25),
        minimum=step_range[0],
        maximum=step_range[1]
    )
    guidance_update = gr.update(
        value=m.get('default_guidance', 7.5),
        minimum=guidance_range[0],
        maximum=guidance_range[1]
    )
    width_update = gr.update(value=m.get('default_width', 512))
    height_update = gr.update(value=m.get('default_height', 512))
    negative_update = gr.update(value=m.get('default_negative', ''))
    
    if model_key in loaded_image_models:
        # already loaded -> hide load button, show unload, enable generate
        return info, gr.update(visible=False), gr.update(visible=True), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=True)
    else:
        # not loaded -> show load button, hide unload, disable generate
        return info, gr.update(visible=True), gr.update(visible=False), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=False)


def load_preview_image(path):
    if not path:
        return None, ""
    try:
        from PIL import Image
        img = Image.open(path)
        return img, f"📂 {os.path.basename(path)}"
    except Exception as e:
        return None, f"❌ Error loading preview: {e}"


def delete_generated_image(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return get_generated_image_choices(), None, f"🗑️ Deleted {os.path.basename(path)}"
        return get_generated_image_choices(), None, "ℹ️ File not found"
    except Exception as e:
        return get_generated_image_choices(), None, f"❌ Error deleting: {e}"


def stop_image_generation():
    global image_gen_cancel_event
    image_gen_cancel_event = True
    return "🛑 Stop requested"


def set_generating_state():
    """Called when generation starts - disable generate, show stop."""
    return gr.update(interactive=False), gr.update(visible=True)


def clear_generating_state():
    """Called when generation ends - enable generate, hide stop."""
    return gr.update(interactive=True), gr.update(visible=False)


def chat(message, history, model_selection, ai_character_selection, user_persona_selection):
    """Handle chat."""
    global current_conversation_id, db
    
    if not message.strip():
        return history
    
    # Extract model key from selection (remove status icon)
    model_key = extract_model_key(model_selection)
    
    try:
        # Create new conversation if needed
        if current_conversation_id is None:
            ai_char_id = None
            user_persona_id = None
            
            if ai_character_selection != "None":
                ai_characters = db.get_all_ai_characters()
                for char in ai_characters:
                    if char['name'] == ai_character_selection:
                        ai_char_id = char['id']
                        break
            
            if user_persona_selection != "None":
                user_personas = db.get_all_user_personas()
                for persona in user_personas:
                    if persona['name'] == user_persona_selection:
                        user_persona_id = persona['id']
                        break
            
            current_conversation_id = db.create_conversation(
                model=model_key,
                ai_character_id=ai_char_id,
                user_persona_id=user_persona_id
            )
            logger.info(f"Started new conversation {current_conversation_id}")
        
        model, tokenizer, chatbot = load_model(model_key)
        
        # Keep only last 3 exchanges to prevent context confusion
        recent_history = history[-3:] if len(history) > 3 else history
        
        # Get AI character details
        system_instruction = None
        char_name = None
        if ai_character_selection != "None":
            ai_characters = db.get_all_ai_characters()
            for char in ai_characters:
                if char['name'] == ai_character_selection:
                    system_instruction = char.get('system_prompt')
                    char_name = char['name']
                    break
        
        # Get user persona context
        user_persona_context = None
        if user_persona_selection != "None":
            user_personas = db.get_all_user_personas()
            for persona in user_personas:
                if persona['name'] == user_persona_selection:
                    context_parts = [f"The user is roleplaying as {persona['name']}."]
                    if persona['description']:
                        context_parts.append(persona['description'])
                    if persona['background']:
                        context_parts.append(persona['background'])
                    user_persona_context = " ".join(context_parts)
                    break
        
        # Format prompt using model-specific template
        prompt = format_prompt_for_model(
            tokenizer=tokenizer,
            model_key=model_selection,
            system_instruction=system_instruction,
            char_name=char_name,
            user_persona_context=user_persona_context,
            recent_history=recent_history,
            current_message=message
        )
        
        logger.info(f"User message: {message}")
        logger.info(f"Model: {model_selection}")
        logger.info(f"Using {len(recent_history)} previous exchanges")
        logger.info(f"AI Character: {ai_character_selection}")
        logger.info(f"System Instruction Length: {len(system_instruction) if system_instruction else 0} chars")
        logger.info(f"Full prompt:\n{prompt}\n---")
        
        # Tokenize with proper limits
        # Use higher limit for character roleplay (long system prompts)
        # truncation_side='left' keeps the most recent conversation (at the end)
        tokenizer.truncation_side = 'left'
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,  # Much higher limit for character roleplay
            return_attention_mask=True
        )
        inputs = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)
        
        logger.info(f"Input tokens: {inputs.shape[1]}")
        
        # Get generation parameters from AI character if selected
        temperature = 0.7
        top_p = 0.9
        top_k = 50
        
        if ai_character_selection != "None":
            ai_characters = db.get_all_ai_characters()
            for char in ai_characters:
                if char['name'] == ai_character_selection:
                    temperature = char.get('temperature', 0.7)
                    top_p = char.get('top_p', 0.9)
                    top_k = char.get('top_k', 50)
                    break
        
        import time
        gen_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=150,  # Limit response length
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.2,  # Higher to prevent repetition
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        gen_elapsed_time = time.time() - gen_start_time
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Model-specific response cleaning
        stop_phrases = []
        
        # Common stop phrases for all models
        stop_phrases.extend(["\nUser:", "\nAssistant:", "User:", "Assistant:"])
        
        # Model-specific stop phrases
        if "Mistral" in model_selection or "Hermes" in model_selection or "Kunoichi" in model_selection or "Dolphin" in model_selection:
            stop_phrases.extend(["[INST]", "[/INST]", "</s>"])
        elif "Vicuna" in model_selection:
            stop_phrases.extend(["USER:", "ASSISTANT:"])
        elif "Llama" in model_selection or "Champion" in model_selection:
            stop_phrases.extend(["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"])
        elif "DeepSeek" in model_selection:
            stop_phrases.extend(["<｜end▁of▁sentence｜>"])
        
        # Add character name stop phrases
        if char_name:
            stop_phrases.extend([f"\n{char_name}:", f"{char_name}:"])
        
        # Clean up response
        for stop in stop_phrases:
            if stop in response:
                response = response.split(stop)[0].strip()
        
        # Remove multiple newlines
        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")
        
        # Remove incomplete sentences at the end (optional, can be disabled)
        if response and len(response) > 20 and not response[-1] in '.!?"\')':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        logger.info(f"Generated response in {gen_elapsed_time:.2f}s: {response}\n---")
        
        # Add timing info to response
        response_with_timing = f"{response}\n\n⏱️ *Generated in {gen_elapsed_time:.2f}s*"
        
        # Save to database (without timing info)
        db.add_message(current_conversation_id, "user", message)
        db.add_message(current_conversation_id, "assistant", response)
        
        history.append((message, response_with_timing))
        return history
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        history.append((message, f"❌ Error: {str(e)}"))
        return history


def clear_chat(model_selection):
    """Clear history and start new conversation."""
    global current_conversation_id
    current_conversation_id = None
    return []


def new_chat():
    """Start a new chat."""
    global current_conversation_id
    current_conversation_id = None
    return [], ""


def load_conversation(conv_title):
    """Load a conversation from database."""
    global current_conversation_id, db
    
    if not conv_title:
        return [], ""
    
    # Find conversation ID from title
    recent_convs = db.get_recent_conversations(20)
    for c in recent_convs:
        if f"{c['title']} ({c['model']})" == conv_title:
            current_conversation_id = c['id']
            history = db.get_conversation_messages(c['id'])
            return history, f"✅ Loaded: {c['title']}"
    
    return [], "❌ Conversation not found"


def export_conversation():
    """Export current conversation as JSON file for download."""
    global current_conversation_id, db
    
    if current_conversation_id is None:
        return None, "❌ No active conversation to export"
    
    try:
        import json
        import tempfile
        import os
        
        data = db.export_conversation(current_conversation_id)
        
        # Create filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in data['title'])
        filename = f"chat_{current_conversation_id}_{safe_title}.json"
        
        # Save to temp file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath, f"✅ Downloading {filename}"
    except Exception as e:
        return None, f"❌ Export failed: {str(e)}"


def delete_conversation():
    """Delete current conversation."""
    global current_conversation_id, db
    
    if current_conversation_id is None:
        return [], "❌ No active conversation to delete"
    
    try:
        db.delete_conversation(current_conversation_id)
        current_conversation_id = None
        return [], "✅ Conversation deleted"
    except Exception as e:
        return [], f"❌ Delete failed: {str(e)}"


def update_chat_model_controls(model_name):
    """Return model info and button visibility based on loaded state."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_MODELS:
        return (
            "*Select a model*",
            gr.update(visible=True),   # load button
            gr.update(visible=False),  # unload button
        )
    
    m = AVAILABLE_MODELS[model_key]
    info = f"*{m['description']}*\n\n**VRAM:** {m['vram']}"
    
    if model_key in active_models:
        # Model is loaded -> hide load, show unload
        return info, gr.update(visible=False), gr.update(visible=True)
    else:
        # Model not loaded -> show load, hide unload
        return info, gr.update(visible=True), gr.update(visible=False)


def update_global_chat_controls(model_name):
    """Return button visibility for the global chat model controls."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_MODELS:
        return gr.update(visible=True), gr.update(visible=False)
    
    if model_key in active_models:
        # Model is loaded -> hide load, show unload
        return gr.update(visible=False), gr.update(visible=True)
    else:
        # Model not loaded -> show load, hide unload
        return gr.update(visible=True), gr.update(visible=False)


def get_chat_model_info(model_name):
    """Get formatted info string for a chat model."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_MODELS:
        return "💡 Select a model to see details"
    
    m = AVAILABLE_MODELS[model_key]
    
    # Determine status
    if model_key in active_models:
        status = "🟢 **LOADED** - Ready!"
    elif is_model_downloaded(m.get('id', ''), is_image_model=False):
        status = "🟡 **Downloaded** - Click Load to use"
    else:
        status = "🔴 **Not Downloaded** - Will download on first load"
    
    desc = m.get('description', '')
    vram = m.get('vram', 'Unknown')
    best_for = m.get('best_for', '')
    
    info = f"**🧠 Text: {model_key}** | {status}\n\n"
    if desc:
        info += f"{desc} "
    if vram:
        info += f"• **VRAM:** {vram}"
    if best_for:
        info += f" • **Best for:** {best_for}"
    
    return info


def get_image_model_info(model_name):
    """Get formatted info string for an image model."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_IMAGE_MODELS:
        return "💡 Select a model to see details"
    
    m = AVAILABLE_IMAGE_MODELS[model_key]
    
    # Determine status
    if model_key in loaded_image_models:
        status = "🟢 **LOADED** - Ready to generate!"
    elif is_model_downloaded(m.get('id', ''), is_image_model=True):
        status = "🟡 **Downloaded** - Click Load to use"
    else:
        status = "🔴 **Not Downloaded** - Will download on first load"
    
    desc = m.get('description', '')
    model_type = m.get('type', 'SD 1.5')
    
    info = f"**🎨 {model_key}** | {status}\n\n"
    if desc:
        info += f"{desc} "
    info += f"• **Type:** {model_type}"
    
    return info


def update_global_img_controls(model_name):
    """Return button visibility for the global image model controls."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_IMAGE_MODELS:
        return gr.update(visible=True), gr.update(visible=False)
    
    if model_key in loaded_image_models:
        # Model is loaded -> hide load, show unload
        return gr.update(visible=False), gr.update(visible=True)
    else:
        # Model not loaded -> show load, hide unload
        return gr.update(visible=True), gr.update(visible=False)


def get_global_status():
    """Get current model status for the global status display with full details."""
    chat_loaded = list(active_models.keys())
    img_loaded = list(loaded_image_models.keys())
    
    if chat_loaded:
        # Return full chat model info
        return get_chat_model_info(chat_loaded[0])
    elif img_loaded:
        # Return full image model info
        return get_image_model_info(img_loaded[0])
    else:
        return "💡 *Select and load a model to start*"


def unload_all_chat_models():
    """Unload all chat models to free VRAM."""
    global active_models
    for key in list(active_models.keys()):
        try:
            del active_models[key]
        except:
            pass
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_image_models():
    """Unload all image models to free VRAM."""
    global image_model, loaded_image_models, current_image_model_key
    for key in list(loaded_image_models.keys()):
        try:
            del loaded_image_models[key]
        except:
            pass
    image_model = None
    current_image_model_key = None
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_chat_model_exclusive(model_name, progress=gr.Progress()):
    """Load a chat model, first unloading any image models."""
    # Unload image models first
    if loaded_image_models:
        logger.info("Unloading image models to free VRAM for chat model...")
        unload_all_image_models()
    
    # Now load the chat model
    return load_chat_model(model_name, progress)


def load_image_model_exclusive(model_name):
    """Load an image model, first unloading any chat models."""
    # Unload chat models first
    if active_models:
        logger.info("Unloading chat models to free VRAM for image model...")
        unload_all_chat_models()
    
    # Now load the image model (let it handle its own progress)
    return load_image_model(model_name)


def load_chat_model(model_name, progress=gr.Progress()):
    """Explicitly load a chat model."""
    model_key = extract_model_key(model_name)
    
    if model_key not in AVAILABLE_MODELS:
        return f"### ❌ Model '{model_key}' not found"
    
    try:
        progress(0.05, desc=f"🔍 Preparing to load {model_key}...")
        model, tokenizer, chatbot = load_model(model_key, progress)
        # Return full model info with loaded status
        return get_chat_model_info(model_name)
    except Exception as e:
        logger.error(f"Failed to load {model_key}: {e}")
        return f"### ❌ Failed to load: {str(e)}"


def start_chat_loading(model_name):
    """Show loading state when chat model load begins."""
    model_key = extract_model_key(model_name)
    return f"### ⏳ Loading {model_key}... Please wait"


def start_image_loading(model_name):
    """Show loading state when image model load begins."""
    model_key = extract_model_key(model_name)
    return f"### ⏳ Loading {model_key}... Please wait"


def unload(model_selection):
    """Unload chat model."""
    model_key = extract_model_key(model_selection)
    if model_key in active_models:
        unload_all_chat_models()
        logger.info(f"Unloaded chat model: {model_key}")
    return get_status()


def cancel_download():
    """Cancel ongoing download."""
    global download_canceled
    download_canceled = True
    return "🛑 Canceling download..."


def create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Create a new AI character."""
    global db
    
    if not name.strip():
        return {
            "status": "❌ Name is required",
            "char_list": list_ai_characters(),
            "name": name,
            "system": system_prompt,
            "desc": description,
            "temp": temperature,
            "topp": top_p,
            "topk": top_k,
            "avatar": avatar
        }
    
    try:
        db.create_ai_character(
            name=name,
            system_prompt=system_prompt,
            description=description,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            avatar=avatar
        )
        return {
            "status": f"✅ Created AI character: {name}",
            "char_list": list_ai_characters(),
            "name": "",
            "system": "",
            "desc": "",
            "temp": 0.7,
            "topp": 0.95,
            "topk": 50,
            "avatar": "🤖"
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "char_list": list_ai_characters(),
            "name": name,
            "system": system_prompt,
            "desc": description,
            "temp": temperature,
            "topp": top_p,
            "topk": top_k,
            "avatar": avatar
        }


def create_user_persona(name, description, background, avatar):
    """Create a new user persona."""
    global db
    
    if not name.strip():
        return {
            "status": "❌ Name is required",
            "persona_list": list_user_personas(),
            "name": name,
            "desc": description,
            "bg": background,
            "avatar": avatar
        }
    
    try:
        db.create_user_persona(
            name=name,
            description=description,
            background=background,
            avatar=avatar
        )
        return {
            "status": f"✅ Created user persona: {name}",
            "persona_list": list_user_personas(),
            "name": "",
            "desc": "",
            "bg": "",
            "avatar": "👤"
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "persona_list": list_user_personas(),
            "name": name,
            "desc": description,
            "bg": background,
            "avatar": avatar
        }


def list_ai_characters():
    """List all AI characters."""
    global db
    
    characters = db.get_all_ai_characters()
    if not characters:
        return "No AI characters yet."
    
    output = []
    for char in characters:
        output.append(f"**{char['avatar']} {char['name']}**")
        if char['description']:
            output.append(f"_{char['description']}_")
        output.append(f"Temperature: {char['temperature']}, Top-P: {char['top_p']}, Top-K: {char['top_k']}")
        if char['system_prompt']:
            output.append(f"Prompt: {char['system_prompt'][:100]}...")
        output.append("---")
    
    return "\n".join(output)


def list_user_personas():
    """List all user personas."""
    global db
    
    personas = db.get_all_user_personas()
    if not personas:
        return "No user personas yet."
    
    output = []
    for persona in personas:
        output.append(f"**{persona['avatar']} {persona['name']}**")
        if persona['description']:
            output.append(f"_{persona['description']}_")
        if persona['background']:
            output.append(f"Background: {persona['background'][:100]}...")
        output.append("---")
    
    return "\n".join(output)


def create_ai_character_tiles():
    """Create tile/card layout for AI characters with clickable navigation."""
    global db
    
    characters = db.get_all_ai_characters()
    
    # Generate tile HTML with direct Gradio integration instructions
    tiles_html = """
    <div style="padding: 10px;">
        <p style="color: #aaa; font-size: 14px; margin-bottom: 15px;">
            💡 <strong>To edit a character:</strong> Select it from the dropdown in the "Edit Character" tab
        </p>
    """
    
    tiles_html += """
    <style>
        .character-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 10px;
        }
        .character-tile {
            border: 2px solid #444;
            border-radius: 12px;
            padding: 15px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        }
        .character-tile:hover {
            border-color: #00a8ff;
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0, 168, 255, 0.3);
        }
        .character-tile.create-new {
            border: 2px dashed #666;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        }
        .character-tile.create-new:hover {
            border-color: #00ff00;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.2);
        }
        .tile-avatar {
            font-size: 48px;
            text-align: center;
            margin-bottom: 10px;
        }
        .tile-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #fff;
            text-align: center;
        }
        .tile-desc {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 10px;
            text-align: center;
            min-height: 40px;
        }
        .tile-params {
            font-size: 12px;
            color: #888;
            text-align: center;
            border-top: 1px solid #444;
            padding-top: 8px;
            margin-top: 8px;
        }
        .create-icon {
            font-size: 64px;
            color: #666;
            text-align: center;
        }
        .create-text {
            font-size: 18px;
            color: #888;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    <div class="character-grid">
    """
    
    # Create New tile
    tiles_html += """
        <div class="character-tile create-new">
            <div>
                <div class="create-icon">➕</div>
                <div class="create-text">Create New Character</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">Go to Edit Character tab →</div>
            </div>
        </div>
    """
    
    for char in characters:
        desc = char['description'] if char['description'] else "No description"
        tiles_html += f"""
        <div class="character-tile">
            <div class="tile-avatar">{char['avatar']}</div>
            <div class="tile-name">{char['name']}</div>
            <div class="tile-desc">{desc}</div>
            <div class="tile-params">
                🌡️ {char['temperature']} | 🎯 P:{char['top_p']} K:{char['top_k']}
            </div>
        </div>
        """
    
    tiles_html += """
    </div>
    </div>
    """
    
    return tiles_html


def create_user_persona_tiles():
    """Create tile/card layout for user personas."""
    global db
    
    personas = db.get_all_user_personas()
    
    tiles_html = """
    <div style="padding: 10px;">
        <p style="color: #aaa; font-size: 14px; margin-bottom: 15px;">
            💡 <strong>To edit a persona:</strong> Select it from the dropdown in the "Edit Persona" tab
        </p>
    """
    
    tiles_html += """
    <style>
        .persona-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 10px;
        }
        .persona-tile {
            border: 2px solid #444;
            border-radius: 12px;
            padding: 15px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        }
        .persona-tile:hover {
            border-color: #ff6b6b;
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3);
        }
        .persona-tile.create-new {
            border: 2px dashed #666;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        }
        .persona-tile.create-new:hover {
            border-color: #00ff00;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.2);
        }
    </style>
    <div class="persona-grid">
    """
    
    # Create New tile
    tiles_html += """
        <div class="persona-tile create-new">
            <div>
                <div class="create-icon">➕</div>
                <div class="create-text">Create New Persona</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">Go to Edit Persona tab →</div>
            </div>
        </div>
    """
    
    for persona in personas:
        desc = persona['description'] if persona['description'] else "No description"
        bg_preview = persona['background'][:80] + "..." if persona['background'] and len(persona['background']) > 80 else (persona['background'] or "No background")
        tiles_html += f"""
        <div class="persona-tile">
            <div class="tile-avatar">{persona['avatar']}</div>
            <div class="tile-name">{persona['name']}</div>
            <div class="tile-desc">{desc}</div>
            <div class="tile-params" style="font-size: 11px;">{bg_preview}</div>
        </div>
        """
    
    tiles_html += """
    </div>
    </div>
    """
    
    return tiles_html



def refresh_dropdowns():
    """Refresh AI character and user persona dropdowns."""
    global db
    
    ai_characters = db.get_all_ai_characters()
    ai_char_names = ["None"] + [char['name'] for char in ai_characters]
    
    user_personas = db.get_all_user_personas()
    user_persona_names = ["None"] + [p['name'] for p in user_personas]
    
    return gr.update(choices=ai_char_names), gr.update(choices=user_persona_names)


def load_ai_character_for_edit(char_name):
    """Load AI character data into form fields for editing."""
    global db
    
    if char_name == "[Create New]":
        return ("", "", "", "", 0.7, 0.95, 50, "🤖")
    
    ai_characters = db.get_all_ai_characters()
    for char in ai_characters:
        if char['name'] == char_name:
            return (
                str(char['id']),                    # hidden ID field
                char['name'],                       # name
                char['system_prompt'] or "",        # system prompt
                char['description'] or "",          # description
                char['temperature'],                # temperature
                char['top_p'],                      # top_p
                char['top_k'],                      # top_k
                char['avatar'] or "🤖"             # avatar
            )
    
    return ("", "", "", "", 0.7, 0.95, 50, "🤖")


def load_user_persona_for_edit(persona_name):
    """Load user persona data into form fields for editing."""
    global db
    
    if persona_name == "[Create New]":
        return ("", "", "", "", "👤")
    
    user_personas = db.get_all_user_personas()
    for persona in user_personas:
        if persona['name'] == persona_name:
            return (
                str(persona['id']),                 # hidden ID field
                persona['name'],                    # name
                persona['description'] or "",       # description
                persona['background'] or "",        # background
                persona['avatar'] or "👤"          # avatar
            )
    
    return ("", "", "", "", "👤")


def save_ai_character(char_id, name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Save AI character (create new or update existing)."""
    global db
    
    if not name.strip():
        return "❌ Name is required"
    
    try:
        if char_id and char_id.strip():
            # Update existing character
            db.update_ai_character(
                int(char_id),
                name=name,
                system_prompt=system_prompt,
                description=description,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                avatar=avatar
            )
            return f"✅ Updated AI character: {name}"
        else:
            # Create new character
            db.create_ai_character(
                name=name,
                system_prompt=system_prompt,
                description=description,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                avatar=avatar
            )
            return f"✅ Created AI character: {name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_user_persona(persona_id, name, description, background, avatar):
    """Save user persona (create new or update existing)."""
    global db
    
    if not name.strip():
        return "❌ Name is required"
    
    try:
        if persona_id and persona_id.strip():
            # Update existing persona
            db.update_user_persona(
                int(persona_id),
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Updated user persona: {name}"
        else:
            # Create new persona
            db.create_user_persona(
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Created user persona: {name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
        return "❌ Name is required", list_user_personas()
    
    try:
        if persona_id and persona_id.strip():
            # Update existing persona
            db.update_user_persona(
                int(persona_id),
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Updated user persona: {name}", list_user_personas()
        else:
            # Create new persona
            db.create_user_persona(
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Created user persona: {name}", list_user_personas()
    except Exception as e:
        return f"❌ Error: {str(e)}", list_user_personas()


def delete_ai_character_by_name(char_id, char_name):
    """Delete AI character by ID."""
    global db
    
    if not char_id or not char_id.strip():
        return "❌ Please select a character to delete"
    
    try:
        db.delete_ai_character(int(char_id))
        return f"✅ Deleted AI character: {char_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_user_persona_by_name(persona_id, persona_name):
    """Delete user persona by ID."""
    global db
    
    if not persona_id or not persona_id.strip():
        return "❌ Please select a persona to delete"
    
    try:
        db.delete_user_persona(int(persona_id))
        return f"✅ Deleted user persona: {persona_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def handle_create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Wrapper to handle AI character creation and return proper outputs."""
    result = create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar)
    
    return (
        result["status"],           # status message
        result["char_list"],        # updated character list
        result["name"],             # cleared/original name
        result["system"],           # cleared/original system prompt
        result["desc"],             # cleared/original description
        result["temp"],             # reset temperature
        result["topp"],             # reset top_p
        result["topk"],             # reset top_k
        result["avatar"]            # reset avatar
    )


def handle_create_user_persona(name, description, background, avatar):
    """Wrapper to handle user persona creation and return proper outputs."""
    result = create_user_persona(name, description, background, avatar)
    
    return (
        result["status"],           # status message
        result["persona_list"],     # updated persona list
        result["name"],             # cleared/original name
        result["desc"],             # cleared/original description
        result["bg"],               # cleared/original background
        result["avatar"]            # reset avatar
    )


def create_ui():
    """Create UI."""
    global db
    
    # Get AI characters and user personas for dropdowns
    ai_characters = db.get_all_ai_characters()
    ai_char_names = ["None"] + [char['name'] for char in ai_characters]
    
    user_personas = db.get_all_user_personas()
    user_persona_names = ["None"] + [p['name'] for p in user_personas]
    
    # Get recent conversations
    recent_convs = db.get_recent_conversations(20)
    conv_choices = {f"{c['title']} ({c['model']})": c['id'] for c in recent_convs}
    
    with gr.Blocks(title="KVGenius Multi-Model Chat", theme=gr.themes.Soft(), css="""
        /* Chat area - larger height */
        #chatbot {
            height: 600px !important;
            overflow-y: auto !important;
        }
        
        /* Side panel scrollable */
        .side-panel {
            overflow-y: auto !important;
            max-height: 700px !important;
        }
        
        /* Loading status bar - prominent styling */
        .loading-status {
            padding: 15px 20px !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            text-align: center !important;
            margin-bottom: 15px !important;
            transition: all 0.3s ease !important;
        }
        
        .loading-status.idle {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
            border: 2px solid #0f3460 !important;
            color: #94a3b8 !important;
        }
        
        .loading-status.loading {
            background: linear-gradient(135deg, #1e3a5f 0%, #0d47a1 100%) !important;
            border: 2px solid #2196f3 !important;
            color: #90caf9 !important;
            animation: pulse-loading 1.5s ease-in-out infinite !important;
        }
        
        .loading-status.success {
            background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%) !important;
            border: 2px solid #40c057 !important;
            color: #8ce99a !important;
        }
        
        .loading-status.error {
            background: linear-gradient(135deg, #4a1a1a 0%, #6b2121 100%) !important;
            border: 2px solid #e74c3c !important;
            color: #ff8a80 !important;
        }
        
        @keyframes pulse-loading {
            0%, 100% { box-shadow: 0 0 10px rgba(33, 150, 243, 0.3); }
            50% { box-shadow: 0 0 25px rgba(33, 150, 243, 0.6); }
        }
        
        /* Tile-style radio buttons */
        .tile-radio label {
            display: grid !important;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
            gap: 15px !important;
            padding: 10px !important;
        }
        
        .tile-radio input[type="radio"] {
            display: none !important;
        }
        
        .tile-radio label > span {
            border: 2px solid #444 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%) !important;
            text-align: center !important;
            font-size: 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100px !important;
        }
        
        /* Style Create New tile with orange color - MUST come after general style */
        .tile-radio label:first-of-type > span {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            border-color: #ff6b35 !important;
        }
        
        .tile-radio label > span:hover {
            border-color: #00a8ff !important;
            transform: translateY(-5px) !important;
            box-shadow: 0 5px 20px rgba(0, 168, 255, 0.3) !important;
        }
        
        /* Keep Create New orange even on hover */
        .tile-radio label:first-of-type > span:hover {
            border-color: #ff8555 !important;
            box-shadow: 0 5px 20px rgba(255, 107, 53, 0.5) !important;
            background: linear-gradient(135deg, #ff8555 0%, #ffa03e 100%) !important;
        }
        
        .tile-radio input[type="radio"]:checked + span {
            border-color: #00ff00 !important;
            background: linear-gradient(135deg, #1e3e1e 0%, #2a4a2a 100%) !important;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.4) !important;
        }
        
        /* Keep Create New orange even when selected */
        .tile-radio input[type="radio"]:first-of-type:checked + span,
        .tile-radio label:first-of-type input[type="radio"]:checked + span {
            border-color: #ff6b35 !important;
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            box-shadow: 0 5px 20px rgba(255, 107, 53, 0.6) !important;
        }
        
        /* Style Create New buttons with distinctive color */
        .create-new-btn {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            color: white !important;
            border: 2px solid #ff6b35 !important;
            font-weight: bold !important;
        }
        
        .create-new-btn:hover {
            background: linear-gradient(135deg, #ff8555 0%, #ffa03e 100%) !important;
            border-color: #ff8555 !important;
            transform: scale(1.05) !important;
        }
        
        /* Compact header styling */
        .header-row {
            align-items: flex-end !important;
            padding-bottom: 10px !important;
        }
        .status-bar {
            font-size: 14px !important;
            padding: 10px 16px !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
            border: 1px solid #0f3460 !important;
            text-align: left !important;
        }
        .status-key {
            font-size: 11px !important;
            color: #888 !important;
            margin-top: 5px !important;
        }
    """) as demo:
        
        # ==================== HEADER WITH INLINE MODEL SELECTORS ====================
        with gr.Row(elem_classes=["header-row"]):
            # Left side - Title
            with gr.Column(scale=1, min_width=200):
                gr.Markdown(f"# 🤖 KVGenius AI Studio\n*{get_gpu_info()}*")
                gr.Markdown("🟢 Loaded | 🟡 Downloaded | 🔴 Not Downloaded", elem_classes=["status-key"])
            
            # Text Model selector with button below
            with gr.Column(scale=1, min_width=220):
                global_chat_dropdown = gr.Dropdown(
                    choices=get_chat_model_choices(),
                    value=get_chat_model_choices()[0] if get_chat_model_choices() else None,
                    label="🧠 Text Model",
                    container=True
                )
                with gr.Row():
                    global_chat_load_btn = gr.Button("📥 Load", variant="primary", size="sm")
                    global_chat_unload_btn = gr.Button("🗑️ Unload", variant="stop", size="sm", visible=False)
                    global_chat_refresh_btn = gr.Button("🔄", size="sm", variant="secondary")
            
            # Image Model selector with button below
            with gr.Column(scale=1, min_width=220):
                global_img_dropdown = gr.Dropdown(
                    choices=get_image_model_choices(),
                    value=get_image_model_choices()[0] if get_image_model_choices() else None,
                    label="🎨 Image Model",
                    container=True
                )
                with gr.Row():
                    global_img_load_btn = gr.Button("📥 Load", variant="primary", size="sm")
                    global_img_unload_btn = gr.Button("🗑️ Unload", variant="stop", size="sm", visible=False)
                    global_img_refresh_btn = gr.Button("🔄", size="sm", variant="secondary")
        
        # Status bar - full width below header (shows model info when selected)
        global_model_status = gr.Markdown(
            "💡 Select a model to see details, then click Load to start",
            elem_classes=["status-bar"]
        )
        
        with gr.Tabs() as main_tabs:
            with gr.TabItem("💬 Chat", id="chat_tab"):
                with gr.Tabs() as chat_tabs:
                    with gr.TabItem("💬 Conversation", id="conversation_tab"):
                        
                        with gr.Row():
                            # Sidebar - Settings (LEFT side)
                            with gr.Column(scale=1):
                                gr.Markdown("### 🎭 Character & Persona")
                                ai_char_dropdown = gr.Dropdown(
                                    choices=ai_char_names,
                                    value="None",
                                    label="AI Character",
                                    info="How the AI behaves"
                                )
                                user_persona_dropdown = gr.Dropdown(
                                    choices=user_persona_names,
                                    value="None",
                                    label="User Persona",
                                    info="Who you are roleplaying as"
                                )
                                
                                gr.Markdown("---")
                                gr.Markdown("### 📂 Conversations")
                                load_conv_dropdown = gr.Dropdown(
                                    choices=list(conv_choices.keys()),
                                    label="Load Conversation"
                                )
                                new_chat_btn = gr.Button("🆕 New Chat", size="sm")
                                
                                with gr.Row():
                                    export_btn = gr.Button("📥 Export", size="sm")
                                    delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                                
                                # Hidden file output for export downloads
                                export_file = gr.File(label="Download", visible=False)
                                
                                gr.Markdown("---")
                                gr.Markdown("### 📊 Status")
                                status = gr.Markdown()
                                
                                with gr.Row():
                                    refresh = gr.Button("🔄 Refresh", size="sm")
                                    cancel_btn = gr.Button("🛑 Cancel DL", size="sm", variant="stop")
                                
                                with gr.Accordion("ℹ️ Help", open=False):
                                    gr.Markdown("""
**Features:**
- 🔄 Switch models anytime
- 💾 Models cached in VRAM
- 🗑️ Unload to free VRAM
- 🎭 AI Characters control AI behavior
- 👤 User Personas define your role

**Privacy:**
- ✅ Runs 100% locally
- ✅ No data sent externally
- ✅ History saved in local database
                                    """)
                            
                            # Main area - Chat (RIGHT side, matching Image Generator)
                            with gr.Column(scale=3):
                                chatbot_ui = gr.Chatbot(label="Chat", show_copy_button=True, elem_id="chatbot", type="tuples", height=500)
                                
                                with gr.Row():
                                    msg = gr.Textbox(
                                        label="Message",
                                        placeholder="Type here and press Enter...",
                                        lines=2,
                                        scale=4
                                    )
                                    send = gr.Button("Send 📤", variant="primary", scale=1)
                                
                                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    
                    with gr.TabItem("👤 Characters", id="char_tab"):
                        # Browse view
                        with gr.Column(visible=True) as char_browse_view:
                            gr.Markdown("## 🎴 Your AI Characters\nClick any character tile to edit it.")
                            
                            # Get character names for radio buttons
                            char_display_names = ["➕ Create New"] + [f"{char['avatar']} {char['name']}" for char in db.get_all_ai_characters()]
                            char_actual_names = ["[Create New]"] + [char['name'] for char in db.get_all_ai_characters()]
                            
                            ai_char_selector = gr.Radio(
                                choices=char_display_names,
                                label="Select Character",
                                elem_classes="tile-radio",
                                container=False
                            )
                            
                            refresh_ai_tiles_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                        
                        # Edit view (hidden by default)
                        with gr.Column(visible=False) as char_edit_view:
                            gr.Markdown("## Edit AI Character\nModify character personality, system prompt, and generation parameters.")
                            
                            back_to_browse_ai_btn = gr.Button("← Back to Gallery", size="sm", variant="secondary")
                            
                            ai_char_id_hidden = gr.Textbox(visible=False, value="")
                            
                            new_ai_name = gr.Textbox(label="Character Name", placeholder="e.g., Friendly Tutor")
                            new_ai_desc = gr.Textbox(label="Short Description", placeholder="Brief description for the dropdown")
                            new_ai_system = gr.Textbox(
                                label="System Prompt (Personality)", 
                                placeholder="You are a helpful tutor who explains things clearly...", 
                                lines=8
                            )
                            
                            gr.Markdown("### Generation Parameters")
                            with gr.Row():
                                new_ai_temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperature", info="Creativity (higher = more random)")
                                new_ai_topp = gr.Slider(0.1, 1.0, value=0.95, label="Top P", info="Nucleus sampling")
                                new_ai_topk = gr.Slider(1, 100, value=50, step=1, label="Top K", info="Token diversity")
                            
                            new_ai_avatar = gr.Textbox(label="Avatar Emoji", value="🤖", max_lines=1)
                            
                            with gr.Row():
                                save_ai_btn = gr.Button("💾 Save Character", variant="primary", size="lg")
                                delete_ai_btn = gr.Button("🗑️ Delete Character", variant="stop", size="lg")
                            
                            ai_create_status = gr.Markdown()
                    
                    with gr.TabItem("🎭 Personas", id="persona_tab"):
                        # Browse view
                        with gr.Column(visible=True) as persona_browse_view:
                            gr.Markdown("## 🎴 Your User Personas\nClick any persona tile to edit it.")
                            
                            # Get persona names for radio buttons
                            persona_display_names = ["➕ Create New"] + [f"{p['avatar']} {p['name']}" for p in db.get_all_user_personas()]
                            persona_actual_names = ["[Create New]"] + [p['name'] for p in db.get_all_user_personas()]
                            
                            user_persona_selector = gr.Radio(
                                choices=persona_display_names,
                                label="Select Persona",
                                elem_classes="tile-radio",
                                container=False
                            )
                            
                            refresh_user_tiles_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                        
                        # Edit view (hidden by default)
                        with gr.Column(visible=False) as persona_edit_view:
                            gr.Markdown("## Edit User Persona\nModify who you are in the roleplay.")
                            
                            back_to_browse_user_btn = gr.Button("← Back to Gallery", size="sm", variant="secondary")
                            
                            user_persona_id_hidden = gr.Textbox(visible=False, value="")
                            
                            new_user_name = gr.Textbox(label="Persona Name", placeholder="e.g., Space Explorer")
                            new_user_desc = gr.Textbox(label="Short Description", placeholder="Brief description (e.g., 'A brave starship captain')")
                            new_user_bg = gr.Textbox(
                                label="Background/Context", 
                                placeholder="Detailed roleplay background: You are Captain Sarah Chen, commanding the starship Odyssey...", 
                                lines=8
                            )
                            new_user_avatar = gr.Textbox(label="Avatar Emoji", value="👤", max_lines=1)
                            
                            with gr.Row():
                                save_user_btn = gr.Button("💾 Save Persona", variant="primary", size="lg")
                                delete_user_btn = gr.Button("🗑️ Delete Persona", variant="stop", size="lg")
                            
                            user_create_status = gr.Markdown()
            
            with gr.TabItem("🎨 Image Generator", id="image_tab"):
                
                with gr.Tabs() as img_tabs:
                    with gr.TabItem("🖌️ Generate", id="generate_tab"):
                        
                        with gr.Row():
                            # Sidebar - Settings
                            with gr.Column(scale=1):
                                gr.Markdown("### 🎭 LoRA")
                                lora_dropdown = gr.Dropdown(
                                    choices=["None"] + get_lora_choices(),
                                    value="None",
                                    label="Select LoRA",
                                    info="Apply a trained LoRA to your generation"
                                )
                                lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA Strength")
                                refresh_lora_btn = gr.Button("🔄 Refresh LoRAs", size="sm")
                                
                                gr.Markdown("---")
                                gr.Markdown("### ⚙️ Settings")
                                img_steps = gr.Slider(1, 50, value=4, step=1, label="Steps", info="SDXL Turbo: 1-4, Others: 20-30")
                                img_guidance = gr.Slider(0, 20, value=0, step=0.5, label="Guidance Scale", info="SDXL Turbo: 0, Others: 7-9")
                                
                                with gr.Row():
                                    img_width = gr.Slider(512, 1024, value=512, step=64, label="Width")
                                    img_height = gr.Slider(512, 1024, value=512, step=64, label="Height")
                                
                                img_seed = gr.Number(value=-1, label="Seed", info="-1 = random")
                            
                            # Main area - Prompts & Output
                            with gr.Column(scale=3):
                                img_prompt = gr.Textbox(
                                    label="✨ Prompt",
                                    placeholder="A majestic dragon flying over a crystal castle at sunset, highly detailed, fantasy art, cinematic lighting",
                                    lines=3
                                )
                                with gr.Row():
                                    img_token_count = gr.Markdown("*0 / 77 tokens*", elem_id="token-counter")
                                    quick_enhance_btn = gr.Button("⚡ Quick Enhance", size="sm", scale=0)
                                    ai_enhance_btn = gr.Button("🧠 AI Enhance", size="sm", scale=0)
                                    save_to_library_btn = gr.Button("💾 Save", size="sm", scale=0)
                                
                                img_negative = gr.Textbox(
                                    label="🚫 Negative Prompt",
                                    placeholder="blurry, bad quality, distorted, ugly, deformed, watermark, text",
                                    lines=2,
                                    value="blurry, bad quality, distorted, ugly, deformed"
                                )
                                
                                with gr.Row():
                                    generate_img_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg", scale=3, interactive=False)
                                    stop_img_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1, visible=False)
                                
                                img_output = gr.Image(label="Generated Image", type="pil", height=512)
                    
                    with gr.TabItem("🖼️ Gallery", id="gallery_tab"):
                        gr.Markdown("### 📁 Your Generated Images\n*Click images in the gallery to select/deselect. Use buttons above for batch operations.*")
                        
                        with gr.Row():
                            refresh_gallery_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                            select_all_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
                            clear_selection_btn = gr.Button("☐ Clear Selection", variant="secondary", size="sm")
                            download_selected_btn = gr.Button("⬇️ Download Selected", variant="primary", size="sm")
                            delete_selected_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
                        
                        gallery_status = gr.Markdown("")
                        img_download_file = gr.File(visible=False)
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                # Gallery grid showing all images
                                img_gallery = gr.Gallery(
                                    value=get_gallery_images(),
                                    label="Click to preview",
                                    columns=4,
                                    rows=3,
                                    height=450,
                                    object_fit="contain",
                                    show_label=False,
                                    allow_preview=True
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("### 📋 Select Images")
                                img_checklist = gr.CheckboxGroup(
                                    choices=get_gallery_checkbox_choices(),
                                    value=[],
                                    label="Select for batch operations",
                                    info="Check images to download or delete multiple"
                                )
                                gr.Markdown("---")
                                gr.Markdown("### 📄 Image Info")
                                img_metadata_display = gr.Markdown("*Click an image to see its generation settings*")
                                load_prompt_from_image_btn = gr.Button("📥 Load Prompt to Generator", variant="secondary", size="sm", visible=False)
                    
                    with gr.TabItem("📚 Prompt Library", id="prompt_library_tab"):
                        gr.Markdown("### 📚 Saved Prompts\n*Save your favorite prompts and settings to reuse them later.*")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("#### 💾 Save Current Settings")
                                prompt_save_name = gr.Textbox(
                                    label="Prompt Name",
                                    placeholder="e.g., Epic Fantasy Portrait",
                                    max_lines=1
                                )
                                save_prompt_btn = gr.Button("💾 Save to Library", variant="primary", size="sm")
                                prompt_library_status = gr.Markdown("")
                            
                            with gr.Column(scale=3):
                                gr.Markdown("#### 📖 Your Saved Prompts")
                                prompt_library_dropdown = gr.Dropdown(
                                    choices=get_prompt_library_choices(),
                                    label="Select a prompt",
                                    interactive=True
                                )
                                with gr.Row():
                                    load_library_prompt_btn = gr.Button("📥 Load to Generator", variant="primary", size="sm")
                                    delete_library_prompt_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                                    refresh_library_btn = gr.Button("🔄", variant="secondary", size="sm")
                        
                        gr.Markdown("---")
                        prompt_preview_display = gr.Markdown("*Select a prompt to see its details*")
                    
                    with gr.TabItem("🎓 LoRA Training", id="lora_training_tab"):
                        gr.Markdown("### 🎓 Train Custom LoRAs\n*Train your own LoRA models to generate specific subjects, styles, or characters.*")
                        
                        with gr.Tabs() as lora_tabs:
                            with gr.TabItem("📁 Dataset", id="dataset_tab"):
                                gr.Markdown("#### Step 1: Prepare Training Images")
                                
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("**Create/Select Dataset**")
                                        new_dataset_name = gr.Textbox(
                                            label="New Dataset Name",
                                            placeholder="e.g., my_alien_species",
                                            max_lines=1
                                        )
                                        create_dataset_btn = gr.Button("➕ Create Dataset", variant="primary", size="sm")
                                        
                                        dataset_dropdown = gr.Dropdown(
                                            choices=get_lora_dataset_choices(),
                                            label="Select Dataset",
                                            interactive=True
                                        )
                                        delete_dataset_btn = gr.Button("🗑️ Delete Dataset", variant="stop", size="sm")
                                        dataset_status = gr.Markdown("")
                                    
                                    with gr.Column(scale=2):
                                        gr.Markdown("**Upload Images**")
                                        training_image_upload = gr.File(
                                            label="Upload Training Images",
                                            file_count="multiple",
                                            file_types=["image"]
                                        )
                                        with gr.Row():
                                            crop_size = gr.Dropdown(
                                                choices=[512, 768, 1024],
                                                value=512,
                                                label="Size"
                                            )
                                            crop_mode = gr.Dropdown(
                                                choices=[
                                                    ("Resize & Pad (keeps all details)", "resize_pad"),
                                                    ("Smart Crop (portrait-aware)", "smart"),
                                                    ("Center Crop", "center"),
                                                    ("Top Crop (for faces)", "top"),
                                                    ("Stretch (may distort)", "resize_stretch")
                                                ],
                                                value="resize_pad",
                                                label="Crop Mode",
                                                info="How to fit non-square images"
                                            )
                                        auto_caption_checkbox = gr.Checkbox(
                                            label="Auto-caption with AI",
                                            value=False,
                                            info="Uses BLIP model (~300MB)"
                                        )
                                        upload_images_btn = gr.Button("📤 Upload & Process", variant="primary")
                                
                                gr.Markdown("---")
                                gr.Markdown("**Dataset Images** *(Click image to edit caption)*")
                                
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        dataset_gallery = gr.Gallery(
                                            label="Training Images",
                                            columns=4,
                                            rows=2,
                                            height=300,
                                            object_fit="cover",
                                            show_label=False
                                        )
                                        with gr.Row():
                                            suggest_captions_btn = gr.Button("🧠 Auto-Caption All", variant="secondary", size="sm")
                                            refresh_dataset_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                                    
                                    with gr.Column(scale=1):
                                        gr.Markdown("**Edit Caption**")
                                        selected_image_name = gr.Textbox(label="Selected Image", interactive=False)
                                        image_caption_edit = gr.Textbox(
                                            label="Caption",
                                            placeholder="Describe what's in this image...",
                                            lines=3
                                        )
                                        with gr.Row():
                                            save_caption_btn = gr.Button("💾 Save Caption", variant="primary", size="sm")
                                            delete_image_btn = gr.Button("🗑️ Delete Image", variant="stop", size="sm")
                                        caption_status = gr.Markdown("")
                            
                            with gr.TabItem("⚙️ Train", id="train_tab"):
                                gr.Markdown("#### Step 2: Configure & Start Training")
                                
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("**Training Configuration**")
                                        
                                        train_dataset_dropdown = gr.Dropdown(
                                            choices=get_lora_dataset_choices(),
                                            label="Dataset to Train",
                                            interactive=True
                                        )
                                        train_output_name = gr.Textbox(
                                            label="LoRA Name",
                                            placeholder="e.g., twilek_species",
                                            max_lines=1
                                        )
                                        train_resume_from = gr.Dropdown(
                                            choices=["(New LoRA)"] + get_lora_choices(),
                                            value="(New LoRA)",
                                            label="Continue Training From",
                                            info="Select existing LoRA to add more images/training"
                                        )
                                        train_trigger_word = gr.Textbox(
                                            label="Trigger Word",
                                            placeholder="e.g., twilek",
                                            info="Word to use in prompts to activate this LoRA",
                                            max_lines=1
                                        )
                                        train_description = gr.Textbox(
                                            label="Description (optional)",
                                            placeholder="Star Wars Twi'lek alien species",
                                            max_lines=2
                                        )
                                        train_base_model = gr.Dropdown(
                                            choices=get_base_model_choices(),
                                            value="Dreamshaper 8",
                                            label="Base Model"
                                        )
                                    
                                    with gr.Column(scale=1):
                                        gr.Markdown("**Training Parameters**")
                                        
                                        train_epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                                        train_learning_rate = gr.Number(value=1e-4, label="Learning Rate")
                                        train_lora_rank = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank", info="Higher = more capacity, more VRAM")
                                        train_lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                                        train_resolution = gr.Dropdown(
                                            choices=[512, 768],
                                            value=512,
                                            label="Training Resolution"
                                        )
                                        train_max_steps = gr.Number(
                                            value=0,
                                            label="Max Steps (0 = use epochs)",
                                            info="Override epochs with fixed step count"
                                        )
                                
                                gr.Markdown("---")
                                
                                with gr.Row():
                                    start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", scale=2)
                                    stop_training_btn = gr.Button("⏹ Stop Training", variant="stop", size="lg", scale=1)
                                
                                training_status = gr.Markdown("*Ready to train*")
                                training_progress = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
                                training_details = gr.Markdown("")
                            
                            with gr.TabItem("📦 My LoRAs", id="my_loras_tab"):
                                gr.Markdown("#### Manage LoRAs")
                                
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        lora_manage_dropdown = gr.Dropdown(
                                            choices=get_lora_choices(),
                                            label="Select LoRA",
                                            interactive=True
                                        )
                                        refresh_loras_btn = gr.Button("🔄 Refresh", size="sm")
                                        delete_lora_btn = gr.Button("🗑️ Delete LoRA", variant="stop", size="sm")
                                        lora_manage_status = gr.Markdown("")
                                    
                                    with gr.Column(scale=2):
                                        lora_info_display = gr.Markdown("*Select a LoRA to see details*")
                                
                                gr.Markdown("---")
                                gr.Markdown("#### 📥 Import Pre-built LoRA")
                                gr.Markdown("*Import .safetensors files from CivitAI or other sources*")
                                
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        import_lora_file = gr.File(
                                            label="LoRA File (.safetensors)",
                                            file_types=[".safetensors"],
                                            type="filepath"
                                        )
                                    with gr.Column(scale=1):
                                        import_lora_name = gr.Textbox(
                                            label="LoRA Name",
                                            placeholder="e.g., TwilekStyle",
                                            info="Name to identify this LoRA"
                                        )
                                        import_trigger_word = gr.Textbox(
                                            label="Trigger Word",
                                            placeholder="e.g., twilek",
                                            info="Word to use in prompts (optional)"
                                        )
                                        import_lora_btn = gr.Button("📥 Import LoRA", variant="primary")
                                
                                import_lora_status = gr.Markdown("")
                                
                                gr.Markdown("---")
                                gr.Markdown("""
**How to Use LoRAs:**

1. Load a compatible base model in the Generate tab
2. Select your LoRA from the dropdown
3. Include the **trigger word** in your prompt
4. Adjust LoRA strength (0.5-1.0 recommended)

*Example:* If your trigger word is `twilek`, use prompts like:
> "a twilek standing in a cantina, star wars, detailed"
                                """)
            
            # ==================== CARD GENERATOR TAB ====================
            with gr.TabItem("🃏 Card Generator", id="card_tab"):
                gr.Markdown("### 🃏 Card Generator\n*Generate custom party game cards using the text model loaded above.*")
                
                with gr.Tabs() as card_tabs:
                    with gr.TabItem("🎴 Generate", id="card_generate_tab"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ Settings")
                                
                                cah_topic = gr.Textbox(
                                    label="📝 Topic / Theme",
                                    placeholder="e.g., Office life, Dating apps, Video games...",
                                    lines=2,
                                    info="What should the cards be about?"
                                )
                                
                                cah_card_type = gr.Dropdown(
                                    choices=get_cah_type_choices(),
                                    value="both",
                                    label="🃏 Card Type"
                                )
                                
                                cah_quantity = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="📊 Quantity",
                                    info="Number of cards to generate"
                                )
                                
                                cah_style = gr.Dropdown(
                                    choices=get_cah_style_choices(),
                                    value="classic",
                                    label="🎨 Style"
                                )
                                
                                cah_custom_style = gr.Textbox(
                                    label="✏️ Custom Style Instructions",
                                    placeholder="Describe your custom style...",
                                    lines=2,
                                    visible=False
                                )
                                
                                generate_cards_btn = gr.Button("🎴 Generate Cards", variant="primary", size="lg")
                                cah_status = gr.Markdown("")
                                
                                gr.Markdown("---")
                                gr.Markdown("""
**💡 Tips:**
- Load a chat model first (Chat tab)
- Be specific with your topic
- Classic style works best for humor
- Try Nerdy for gaming/tech references
- Wholesome for family-friendly cards
                                """)
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### 🎴 Generated Cards")
                                cah_output = gr.Dataframe(
                                    headers=["Generated Cards"],
                                    datatype=["str"],
                                    col_count=(1, "fixed"),
                                    row_count=(10, "dynamic"),
                                    wrap=True,
                                    interactive=False
                                )
                    
                    with gr.TabItem("📰 From Article", id="card_article_tab"):
                        gr.Markdown("### 📰 Extract Cards from Articles\n*Paste an article or enter a URL (like from Clickhole, The Onion, etc.) and extract the funniest lines as CAH cards.*")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ Settings")
                                
                                # Toggle between URL and Text input
                                article_input_mode = gr.Radio(
                                    choices=["📄 Paste Text", "🔗 Enter URL"],
                                    value="📄 Paste Text",
                                    label="Input Mode",
                                    interactive=True
                                )
                                
                                # URL input (initially hidden)
                                article_url_input = gr.Textbox(
                                    label="🔗 Article URL",
                                    placeholder="https://clickhole.com/article-name...",
                                    lines=1,
                                    visible=False,
                                    info="Enter a URL to fetch article text automatically"
                                )
                                fetch_url_btn = gr.Button("🔗 Fetch Article", variant="secondary", size="sm", visible=False)
                                fetch_status = gr.Markdown("", visible=False)
                                
                                # Text input (initially visible)
                                article_input = gr.Textbox(
                                    label="📄 Article Text",
                                    placeholder="Paste the full article text here...\n\nThe AI will extract the funniest jokes and phrases, adjusting verb tenses to work as CAH cards.",
                                    lines=12,
                                    max_lines=20,
                                    visible=True,
                                    info="Paste articles from comedy sites like Clickhole, The Onion, etc."
                                )
                                
                                article_card_type = gr.Dropdown(
                                    choices=get_cah_type_choices(),
                                    value="both",
                                    label="🃏 Card Type"
                                )
                                
                                article_quantity = gr.Slider(
                                    minimum=3,
                                    maximum=15,
                                    value=8,
                                    step=1,
                                    label="📊 Cards to Extract",
                                    info="How many cards to extract from the article"
                                )
                                
                                extract_cards_btn = gr.Button("📰 Extract Cards", variant="primary", size="lg")
                                article_status = gr.Markdown("")
                                
                                gr.Markdown("---")
                                gr.Markdown("""
**💡 How it works:**
- Paste text OR enter a URL to fetch
- AI finds the funniest lines/jokes
- Adjusts verb tenses for CAH format
- Converts to black (prompt) or white (answer) cards
- Great for Clickhole, The Onion, etc.

**Tips:**
- Longer articles = more material
- Articles with absurd premises work best
- Choose "Both" for variety
                                """)
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### 🎴 Extracted Cards")
                                article_output = gr.Dataframe(
                                    headers=["Extracted Cards"],
                                    datatype=["str"],
                                    col_count=(1, "fixed"),
                                    row_count=(10, "dynamic"),
                                    wrap=True,
                                    interactive=False
                                )
                    
                    with gr.TabItem("📚 My Cards", id="card_library_tab"):
                        gr.Markdown("### 📚 Saved Cards\n*All your generated cards are automatically saved here.*")
                        
                        with gr.Row():
                            refresh_cards_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                            export_json_btn = gr.Button("📥 Export JSON", variant="secondary", size="sm")
                            export_txt_btn = gr.Button("📥 Export TXT", variant="secondary", size="sm")
                            export_csv_btn = gr.Button("📥 Export CSV", variant="secondary", size="sm")
                            clear_cards_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                        
                        cah_export_file = gr.File(visible=False)
                        cah_library_status = gr.Markdown("")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                cah_library_list = gr.Dataframe(
                                    value=[[card] for card in get_saved_cards_display()],
                                    headers=["Saved Cards"],
                                    datatype=["str"],
                                    col_count=(1, "fixed"),
                                    row_count=(15, "dynamic"),
                                    wrap=True,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1):
                                cah_stats_display = gr.Markdown(get_cah_stats())
                                
                                gr.Markdown("---")
                                gr.Markdown("### 🗑️ Delete Cards")
                                
                                # Quick delete last N cards
                                with gr.Row():
                                    delete_last_n = gr.Slider(
                                        minimum=1,
                                        maximum=20,
                                        value=1,
                                        step=1,
                                        label="Delete last N cards"
                                    )
                                    delete_last_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                                
                                gr.Markdown("---")
                                
                                # Select specific cards to delete
                                delete_card_select = gr.Dropdown(
                                    choices=get_saved_cards_with_ids(),
                                    label="Select cards to delete",
                                    multiselect=True,
                                    interactive=True
                                )
                                delete_selected_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
                
                # Card Generator Events
                def toggle_custom_style(style):
                    return gr.update(visible=(style == "custom"))
                
                cah_style.change(toggle_custom_style, inputs=cah_style, outputs=cah_custom_style)
                
                generate_cards_btn.click(
                    generate_cah_cards,
                    inputs=[cah_topic, cah_card_type, cah_quantity, cah_style, cah_custom_style],
                    outputs=[cah_output, cah_status]
                ).then(
                    refresh_cah_cards,
                    outputs=[cah_library_list, cah_stats_display]
                )
                
                # Article Extraction Events
                
                # Toggle between URL and text input modes
                def toggle_article_input_mode(mode):
                    if mode == "🔗 Enter URL":
                        return (
                            gr.update(visible=True),   # URL input
                            gr.update(visible=True),   # Fetch button
                            gr.update(visible=True),   # Fetch status
                            gr.update(visible=False),  # Text input
                        )
                    else:
                        return (
                            gr.update(visible=False),  # URL input
                            gr.update(visible=False),  # Fetch button
                            gr.update(visible=False),  # Fetch status
                            gr.update(visible=True),   # Text input
                        )
                
                article_input_mode.change(
                    toggle_article_input_mode,
                    inputs=article_input_mode,
                    outputs=[article_url_input, fetch_url_btn, fetch_status, article_input]
                )
                
                # Fetch article from URL
                fetch_url_btn.click(
                    fetch_article_from_url,
                    inputs=article_url_input,
                    outputs=[article_input, fetch_status]
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=article_input
                )
                
                extract_cards_btn.click(
                    extract_cah_from_article,
                    inputs=[article_input, article_card_type, article_quantity],
                    outputs=[article_output, article_status]
                ).then(
                    refresh_cah_cards,
                    outputs=[cah_library_list, cah_stats_display]
                )
                
                export_json_btn.click(
                    lambda: export_cah_cards("json"),
                    outputs=[cah_export_file, cah_library_status]
                )
                
                export_txt_btn.click(
                    lambda: export_cah_cards("txt"),
                    outputs=[cah_export_file, cah_library_status]
                )
                
                export_csv_btn.click(
                    lambda: export_cah_cards("csv"),
                    outputs=[cah_export_file, cah_library_status]
                )
                
                clear_cards_btn.click(
                    clear_all_cah_cards,
                    outputs=[cah_library_list, cah_library_status]
                ).then(
                    get_cah_stats,
                    outputs=cah_stats_display
                ).then(
                    lambda: gr.update(choices=[]),
                    outputs=delete_card_select
                )
                
                # Delete last N cards
                delete_last_btn.click(
                    delete_last_n_cards,
                    inputs=delete_last_n,
                    outputs=[cah_library_list, cah_library_status]
                ).then(
                    get_cah_stats,
                    outputs=cah_stats_display
                ).then(
                    lambda: gr.update(choices=get_saved_cards_with_ids()),
                    outputs=delete_card_select
                )
                
                # Delete selected cards
                delete_selected_btn.click(
                    delete_selected_cards,
                    inputs=delete_card_select,
                    outputs=[cah_library_list, cah_library_status]
                ).then(
                    get_cah_stats,
                    outputs=cah_stats_display
                ).then(
                    lambda: gr.update(choices=get_saved_cards_with_ids(), value=[]),
                    outputs=delete_card_select
                )
                
                # Refresh also updates delete dropdown
                refresh_cards_btn.click(
                    refresh_cah_cards,
                    outputs=[cah_library_list, cah_stats_display]
                ).then(
                    lambda: gr.update(choices=get_saved_cards_with_ids()),
                    outputs=delete_card_select
                )
            
            # ==================== LORA MANAGER TAB ====================
            with gr.TabItem("🎛️ LoRA Manager", id="lora_manager_tab"):
                gr.Markdown("### 🎛️ LoRA Manager\n*Browse, manage, and configure LoRAs for image generation and text models.*")
                
                with gr.Row():
                    lora_refresh_btn = gr.Button("🔄 Refresh LoRAs", size="sm")
                    lora_scan_status = gr.Markdown("")
                
                with gr.Tabs() as lora_manager_tabs:
                    # Image LoRAs Section
                    with gr.TabItem("🎨 Image LoRAs", id="image_loras_tab"):
                        gr.Markdown("#### Image Generation LoRAs\n*LoRAs for Stable Diffusion models (SD 1.5, SDXL)*")
                        
                        image_lora_list = gr.Dataframe(
                            headers=["Name", "Base Model", "Trigger Words", "Category"],
                            datatype=["str", "str", "str", "str"],
                            label="Available Image LoRAs",
                            interactive=False,
                            wrap=True,
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("**Selected LoRA Details**")
                                selected_image_lora = gr.Dropdown(
                                    label="Select LoRA to View/Edit",
                                    choices=["None"],
                                    value="None"
                                )
                                image_lora_details = gr.Markdown("*Select a LoRA to see details*")
                            
                            with gr.Column(scale=1):
                                gr.Markdown("**Quick Actions**")
                                image_lora_preview = gr.Image(
                                    label="Preview",
                                    height=150,
                                    show_label=False,
                                    visible=False
                                )
                        
                        with gr.Accordion("📝 Edit Metadata", open=False):
                            edit_image_lora_name = gr.Textbox(label="Name", interactive=True)
                            edit_image_lora_triggers = gr.Textbox(label="Trigger Words (comma-separated)", interactive=True)
                            edit_image_lora_base = gr.Dropdown(
                                label="Base Model",
                                choices=["sd15", "sdxl", "unknown"],
                                interactive=True
                            )
                            edit_image_lora_category = gr.Textbox(label="Category", interactive=True)
                            edit_image_lora_desc = gr.Textbox(label="Description", lines=2, interactive=True)
                            save_image_lora_btn = gr.Button("💾 Save Changes", variant="primary", size="sm")
                            edit_lora_status = gr.Markdown("")
                    
                    # Text Model LoRAs Section
                    with gr.TabItem("💬 Text LoRAs", id="text_loras_tab"):
                        gr.Markdown("#### Text Model LoRAs\n*LoRAs for chat and text generation models (Mistral, Llama, etc.)*")
                        
                        text_lora_list = gr.Dataframe(
                            headers=["Name", "Base Model", "Category"],
                            datatype=["str", "str", "str"],
                            label="Available Text LoRAs",
                            interactive=False,
                        )
                        
                        gr.Markdown("*Text model LoRAs coming soon - place PEFT adapters in `data/lora_models/text/`*")
                    
                    # CAH LoRAs Section
                    with gr.TabItem("🃏 CAH LoRAs", id="cah_loras_tab"):
                        gr.Markdown("#### Card Generation LoRAs\n*Specialized LoRAs for CAH-style card generation*")
                        
                        cah_lora_list = gr.Dataframe(
                            headers=["Name", "Base Model", "Theme"],
                            datatype=["str", "str", "str"],
                            label="Available CAH LoRAs",
                            interactive=False,
                        )
                        
                        gr.Markdown("*CAH LoRAs coming soon - train custom LoRAs for specific card themes*")
                
                # LoRA Manager Events
                def refresh_lora_lists():
                    """Refresh all LoRA lists from disk."""
                    from src.models.lora_manager import get_lora_manager
                    lm = get_lora_manager()
                    lm.scan_all()
                    
                    # Build image LoRA data
                    image_data = []
                    image_choices = ["None"]
                    for name, lora in sorted(lm.image_loras.items()):
                        triggers = ", ".join(lora.trigger_words) if lora.trigger_words else ""
                        image_data.append([name, lora.base_model.value.upper(), triggers, lora.category])
                        image_choices.append(name)
                    
                    # Build text LoRA data
                    text_data = []
                    for name, lora in sorted(lm.text_loras.items()):
                        text_data.append([name, lora.base_model.value.upper(), lora.category])
                    
                    # Build CAH LoRA data
                    cah_data = []
                    for name, lora in sorted(lm.cah_loras.items()):
                        cah_data.append([name, lora.base_model.value.upper(), lora.category])
                    
                    status = f"✅ Found {len(lm.image_loras)} image, {len(lm.text_loras)} text, {len(lm.cah_loras)} CAH LoRAs"
                    
                    return (
                        image_data if image_data else [["No LoRAs found", "-", "-", "-"]],
                        gr.update(choices=image_choices),
                        text_data if text_data else [["No LoRAs found", "-", "-"]],
                        cah_data if cah_data else [["No LoRAs found", "-", "-"]],
                        status
                    )
                
                def get_image_lora_details(lora_name):
                    """Get details for selected image LoRA."""
                    if not lora_name or lora_name == "None":
                        return "*Select a LoRA to see details*", "", "", "unknown", "", "", gr.update(visible=False)
                    
                    from src.models.lora_manager import get_lora_manager
                    lm = get_lora_manager()
                    lora = lm.get_lora_info(lora_name)
                    
                    if not lora:
                        return f"*LoRA '{lora_name}' not found*", "", "", "unknown", "", "", gr.update(visible=False)
                    
                    details = f"""
**Name:** {lora.name}
**Path:** `{lora.path}`
**Base Model:** {lora.base_model.value.upper()}
**Trigger Words:** {', '.join(lora.trigger_words) if lora.trigger_words else 'None'}
**Category:** {lora.category}
**Rank:** {lora.rank}
**Description:** {lora.description or 'No description'}
"""
                    
                    # Check for preview image
                    preview_visible = False
                    if lora.preview_image and os.path.exists(lora.preview_image):
                        preview_visible = True
                    
                    return (
                        details,
                        lora.name,
                        ", ".join(lora.trigger_words),
                        lora.base_model.value,
                        lora.category,
                        lora.description,
                        gr.update(visible=preview_visible, value=lora.preview_image if preview_visible else None)
                    )
                
                def save_image_lora_metadata(lora_name, new_name, triggers, base_model, category, description):
                    """Save updated LoRA metadata."""
                    if not lora_name or lora_name == "None":
                        return "❌ Select a LoRA first"
                    
                    from src.models.lora_manager import get_lora_manager, LoRABaseModel
                    lm = get_lora_manager()
                    
                    trigger_list = [t.strip() for t in triggers.split(",") if t.strip()]
                    
                    success = lm.update_lora_metadata(lora_name, {
                        "name": new_name,
                        "trigger_words": trigger_list,
                        "base_model": LoRABaseModel(base_model),
                        "category": category,
                        "description": description,
                    })
                    
                    if success:
                        return f"✅ Saved metadata for {new_name}"
                    else:
                        return f"❌ Failed to save metadata"
                
                # Wire up events
                lora_refresh_btn.click(
                    refresh_lora_lists,
                    outputs=[image_lora_list, selected_image_lora, text_lora_list, cah_lora_list, lora_scan_status]
                )
                
                selected_image_lora.change(
                    get_image_lora_details,
                    inputs=selected_image_lora,
                    outputs=[image_lora_details, edit_image_lora_name, edit_image_lora_triggers, edit_image_lora_base, edit_image_lora_category, edit_image_lora_desc, image_lora_preview]
                )
                
                save_image_lora_btn.click(
                    save_image_lora_metadata,
                    inputs=[selected_image_lora, edit_image_lora_name, edit_image_lora_triggers, edit_image_lora_base, edit_image_lora_category, edit_image_lora_desc],
                    outputs=edit_lora_status
                ).then(
                    refresh_lora_lists,
                    outputs=[image_lora_list, selected_image_lora, text_lora_list, cah_lora_list, lora_scan_status]
                )
                
                # Auto-refresh on tab load
                demo.load(
                    refresh_lora_lists,
                    outputs=[image_lora_list, selected_image_lora, text_lora_list, cah_lora_list, lora_scan_status]
                )
        
        # Chat Events
        # Model controls - outputs for button visibility + status
        chat_model_outputs = [global_chat_load_btn, global_chat_unload_btn, global_model_status]
        
        def update_global_chat_controls_with_info(model_name):
            """Return button visibility and model info for status bar."""
            controls = update_global_chat_controls(model_name)
            info = get_chat_model_info(model_name)
            return controls[0], controls[1], info
        
        global_chat_dropdown.change(
            update_global_chat_controls_with_info,
            inputs=global_chat_dropdown, 
            outputs=chat_model_outputs
        )
        
        # Image model controls - outputs for button visibility and slider presets + status
        img_model_outputs = [global_img_load_btn, global_img_unload_btn, img_steps, img_guidance, img_width, img_height, img_negative, generate_img_btn, global_model_status]
        
        def update_global_img_controls_full(model_name):
            """Return button visibility, slider presets, and model info for global image controls."""
            model_key = extract_model_key(model_name)
            info = get_image_model_info(model_name)
            
            if model_key not in AVAILABLE_IMAGE_MODELS:
                return (
                    gr.update(visible=True),   # load button
                    gr.update(visible=False),  # unload button
                    gr.update(),               # steps slider
                    gr.update(),               # guidance slider
                    gr.update(),               # width slider
                    gr.update(),               # height slider
                    gr.update(),               # negative prompt
                    gr.update(interactive=False),  # generate button disabled
                    info                       # status info
                )
            
            m = AVAILABLE_IMAGE_MODELS[model_key]
            step_range = m.get('step_range', [1, 50])
            guidance_range = m.get('guidance_range', [0.0, 20.0])
            
            steps_update = gr.update(value=m.get('default_steps', 25), minimum=step_range[0], maximum=step_range[1])
            guidance_update = gr.update(value=m.get('default_guidance', 7.5), minimum=guidance_range[0], maximum=guidance_range[1])
            width_update = gr.update(value=m.get('default_width', 512))
            height_update = gr.update(value=m.get('default_height', 512))
            negative_update = gr.update(value=m.get('default_negative', ''))
            
            if model_key in loaded_image_models:
                return gr.update(visible=False), gr.update(visible=True), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=True), info
            else:
                return gr.update(visible=True), gr.update(visible=False), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=False), info
        
        global_img_dropdown.change(
            update_global_img_controls_full,
            inputs=global_img_dropdown, 
            outputs=img_model_outputs
        )
        
        # Button-only outputs for .then() chains (no status update)
        chat_btn_outputs = [global_chat_load_btn, global_chat_unload_btn]
        img_btn_outputs = [global_img_load_btn, global_img_unload_btn, img_steps, img_guidance, img_width, img_height, img_negative, generate_img_btn]
        
        def update_global_img_controls_no_status(model_name):
            """Return button visibility and slider presets without status update."""
            model_key = extract_model_key(model_name)
            
            if model_key not in AVAILABLE_IMAGE_MODELS:
                return (
                    gr.update(visible=True),   # load button
                    gr.update(visible=False),  # unload button
                    gr.update(),               # steps slider
                    gr.update(),               # guidance slider
                    gr.update(),               # width slider
                    gr.update(),               # height slider
                    gr.update(),               # negative prompt
                    gr.update(interactive=False)  # generate button disabled
                )
            
            m = AVAILABLE_IMAGE_MODELS[model_key]
            step_range = m.get('step_range', [1, 50])
            guidance_range = m.get('guidance_range', [0.0, 20.0])
            
            steps_update = gr.update(value=m.get('default_steps', 25), minimum=step_range[0], maximum=step_range[1])
            guidance_update = gr.update(value=m.get('default_guidance', 7.5), minimum=guidance_range[0], maximum=guidance_range[1])
            width_update = gr.update(value=m.get('default_width', 512))
            height_update = gr.update(value=m.get('default_height', 512))
            negative_update = gr.update(value=m.get('default_negative', ''))
            
            if model_key in loaded_image_models:
                return gr.update(visible=False), gr.update(visible=True), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=True)
            else:
                return gr.update(visible=True), gr.update(visible=False), steps_update, guidance_update, width_update, height_update, negative_update, gr.update(interactive=False)
        
        # Load chat model (exclusive - unloads image model first)
        global_chat_load_btn.click(
            start_chat_loading, inputs=global_chat_dropdown, outputs=global_model_status
        ).then(
            load_chat_model_exclusive, inputs=global_chat_dropdown, outputs=global_model_status
        ).then(
            refresh_chat_dropdown, inputs=global_chat_dropdown, outputs=global_chat_dropdown
        ).then(
            lambda m: update_global_chat_controls(m), inputs=global_chat_dropdown, outputs=chat_btn_outputs
        ).then(
            lambda: (gr.update(choices=get_image_model_choices()), gr.update(visible=True), gr.update(visible=False)),
            outputs=[global_img_dropdown, global_img_load_btn, global_img_unload_btn]
        ).then(
            get_global_status, outputs=global_model_status
        )
        
        global_chat_unload_btn.click(
            unload, global_chat_dropdown, status
        ).then(
            refresh_chat_dropdown, inputs=global_chat_dropdown, outputs=global_chat_dropdown
        ).then(
            lambda m: update_global_chat_controls(m), inputs=global_chat_dropdown, outputs=chat_btn_outputs
        ).then(
            get_global_status, outputs=global_model_status
        )
        
        # Load image model (exclusive - unloads chat model first)
        global_img_load_btn.click(
            start_image_loading, inputs=global_img_dropdown, outputs=global_model_status
        ).then(
            load_image_model_exclusive, inputs=global_img_dropdown, outputs=global_model_status
        ).then(
            refresh_image_dropdown, inputs=global_img_dropdown, outputs=global_img_dropdown
        ).then(
            update_global_img_controls_no_status, inputs=global_img_dropdown, outputs=img_btn_outputs
        ).then(
            lambda: (gr.update(choices=get_chat_model_choices()), gr.update(visible=True), gr.update(visible=False)),
            outputs=[global_chat_dropdown, global_chat_load_btn, global_chat_unload_btn]
        ).then(
            get_global_status, outputs=global_model_status
        )
        
        global_img_unload_btn.click(
            unload_image_model, inputs=global_img_dropdown, outputs=global_model_status
        ).then(
            refresh_image_dropdown, inputs=global_img_dropdown, outputs=global_img_dropdown
        ).then(
            update_global_img_controls_no_status, inputs=global_img_dropdown, outputs=img_btn_outputs
        ).then(
            get_global_status, outputs=global_model_status
        )
        
        # Separate refresh buttons for each model type
        global_chat_refresh_btn.click(
            lambda: gr.update(choices=get_chat_model_choices()),
            outputs=global_chat_dropdown
        )
        global_img_refresh_btn.click(
            lambda: gr.update(choices=get_image_model_choices()),
            outputs=global_img_dropdown
        )
        
        msg.submit(chat, [msg, chatbot_ui, global_chat_dropdown, ai_char_dropdown, user_persona_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status).then(lambda m: update_global_chat_controls(m), inputs=global_chat_dropdown, outputs=chat_model_outputs)
        
        send.click(chat, [msg, chatbot_ui, global_chat_dropdown, ai_char_dropdown, user_persona_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status).then(lambda m: update_global_chat_controls(m), inputs=global_chat_dropdown, outputs=chat_model_outputs)
        
        clear_btn.click(clear_chat, global_chat_dropdown, chatbot_ui)
        new_chat_btn.click(new_chat, outputs=[chatbot_ui, status])
        
        load_conv_dropdown.change(load_conversation, load_conv_dropdown, [chatbot_ui, status])
        
        export_btn.click(export_conversation, outputs=[export_file, status])
        delete_btn.click(delete_conversation, outputs=[chatbot_ui, status])
        
        refresh.click(get_status, outputs=status)
        cancel_btn.click(cancel_download, outputs=status).then(get_status, outputs=status)
        
        # LoRA refresh button
        refresh_lora_btn.click(lambda: gr.update(choices=["None"] + get_lora_choices()), outputs=lora_dropdown)

        # Live token counter for prompt
        img_prompt.change(update_token_counter, inputs=img_prompt, outputs=img_token_count)

        # Prompt enhancement buttons (auto-detect style from prompt)
        quick_enhance_btn.click(
            quick_enhance_prompt,
            inputs=[img_prompt],
            outputs=[img_prompt, global_model_status]
        ).then(update_token_counter, inputs=img_prompt, outputs=img_token_count)
        
        ai_enhance_btn.click(
            ai_enhance_prompt,
            inputs=[img_prompt],
            outputs=[img_prompt, global_model_status]
        ).then(update_token_counter, inputs=img_prompt, outputs=img_token_count)

        # Save to library button - switches to library tab
        save_to_library_btn.click(
            lambda: gr.Tabs(selected="prompt_library_tab"),
            outputs=img_tabs
        )

        # Generate with button state management
        generate_img_btn.click(
            set_generating_state,
            outputs=[generate_img_btn, stop_img_btn]
        ).then(
            generate_image,
            inputs=[img_prompt, img_negative, img_steps, img_guidance, img_width, img_height, img_seed, lora_dropdown, lora_strength],
            outputs=[img_output, global_model_status]
        ).then(
            clear_generating_state,
            outputs=[generate_img_btn, stop_img_btn]
        )
        stop_img_btn.click(stop_image_generation, outputs=global_model_status)

        # Gallery events
        refresh_gallery_btn.click(refresh_gallery, outputs=[img_gallery, img_checklist, gallery_status])
        select_all_btn.click(select_all_images, outputs=[img_checklist, gallery_status])
        clear_selection_btn.click(clear_image_selection, outputs=[img_checklist, gallery_status])
        delete_selected_btn.click(
            delete_selected_images, 
            inputs=img_checklist, 
            outputs=[img_gallery, img_checklist, gallery_status]
        )
        download_selected_btn.click(
            download_selected_images, 
            inputs=img_checklist, 
            outputs=[img_download_file, gallery_status]
        )
        
        # Gallery image selection - show metadata
        img_gallery.select(on_gallery_select, outputs=[img_metadata_display, load_prompt_from_image_btn])
        
        # Load prompt from selected image
        load_prompt_from_image_btn.click(
            load_prompt_from_gallery_image,
            outputs=[img_prompt, img_negative, img_steps, img_guidance, img_width, img_height, global_model_status]
        )
        
        # Prompt Library events
        save_prompt_btn.click(
            save_current_prompt,
            inputs=[prompt_save_name, img_prompt, img_negative, img_steps, img_guidance, img_width, img_height],
            outputs=[prompt_library_dropdown, prompt_library_status]
        )
        
        load_library_prompt_btn.click(
            load_prompt_from_library,
            inputs=prompt_library_dropdown,
            outputs=[img_prompt, img_negative, img_steps, img_guidance, img_width, img_height, prompt_save_name, prompt_library_status]
        )
        
        delete_library_prompt_btn.click(
            delete_prompt_from_library,
            inputs=prompt_library_dropdown,
            outputs=[prompt_library_dropdown, prompt_library_status]
        )
        
        refresh_library_btn.click(
            lambda: gr.update(choices=get_prompt_library_choices()),
            outputs=prompt_library_dropdown
        )
        
        prompt_library_dropdown.change(
            format_prompt_preview,
            inputs=prompt_library_dropdown,
            outputs=prompt_preview_display
        )
        
        # ==================== LoRA Training Events ====================
        
        # Dataset management
        create_dataset_btn.click(
            create_new_dataset,
            inputs=new_dataset_name,
            outputs=[dataset_status, dataset_dropdown]
        ).then(
            lambda: gr.update(choices=get_lora_dataset_choices()),
            outputs=train_dataset_dropdown
        )
        
        delete_dataset_btn.click(
            delete_dataset,
            inputs=dataset_dropdown,
            outputs=[dataset_status, dataset_dropdown, dataset_gallery]
        ).then(
            lambda: gr.update(choices=get_lora_dataset_choices()),
            outputs=train_dataset_dropdown
        )
        
        # Image upload
        upload_images_btn.click(
            upload_training_images,
            inputs=[training_image_upload, dataset_dropdown, crop_size, crop_mode, auto_caption_checkbox],
            outputs=[dataset_status, dataset_gallery]
        )
        
        # Load dataset images when selected
        dataset_dropdown.change(
            load_dataset_images,
            inputs=dataset_dropdown,
            outputs=dataset_gallery
        )
        
        # Gallery image selection for caption editing
        dataset_gallery.select(
            get_image_caption,
            inputs=dataset_dropdown,
            outputs=[image_caption_edit, selected_image_name]
        )
        
        # Save caption
        save_caption_btn.click(
            update_image_caption,
            inputs=[dataset_dropdown, selected_image_name, image_caption_edit],
            outputs=caption_status
        ).then(
            load_dataset_images,
            inputs=dataset_dropdown,
            outputs=dataset_gallery
        )
        
        # Delete image
        delete_image_btn.click(
            delete_dataset_image,
            inputs=[dataset_dropdown, selected_image_name],
            outputs=[caption_status, dataset_gallery]
        )
        
        # Auto-caption all images
        suggest_captions_btn.click(
            suggest_captions_for_dataset,
            inputs=dataset_dropdown,
            outputs=[dataset_status, dataset_gallery]
        )
        
        # Refresh dataset gallery
        refresh_dataset_btn.click(
            load_dataset_images,
            inputs=dataset_dropdown,
            outputs=dataset_gallery
        )
        
        # Auto-fill when selecting LoRA to continue
        train_resume_from.change(
            on_resume_lora_selected,
            inputs=train_resume_from,
            outputs=[train_output_name, train_trigger_word, train_description]
        )
        
        # Training controls
        start_training_btn.click(
            start_lora_training,
            inputs=[
                train_dataset_dropdown, train_output_name, train_resume_from, train_base_model,
                train_trigger_word, train_description, train_epochs,
                train_learning_rate, train_lora_rank, train_lora_alpha,
                train_resolution, train_max_steps
            ],
            outputs=[training_status, training_details]
        )
        
        stop_training_btn.click(
            stop_training,
            outputs=[training_status, training_progress, training_details]
        )
        
        # LoRA management (My LoRAs tab)
        lora_manage_dropdown.change(
            get_lora_info,
            inputs=lora_manage_dropdown,
            outputs=lora_info_display
        )
        
        refresh_loras_btn.click(
            lambda: gr.update(choices=get_lora_choices()),
            outputs=lora_manage_dropdown
        )
        
        # Delete LoRA button - update BOTH dropdowns
        delete_lora_btn.click(
            delete_lora,
            inputs=lora_manage_dropdown,
            outputs=[lora_manage_status, lora_manage_dropdown]
        ).then(
            lambda: gr.update(choices=["None"] + get_lora_choices()),
            outputs=lora_dropdown
        )
        
        # Import LoRA button - update BOTH dropdowns (manage tab AND generate tab)
        import_lora_btn.click(
            import_lora,
            inputs=[import_lora_file, import_lora_name, import_trigger_word],
            outputs=[import_lora_status, lora_manage_dropdown]
        ).then(
            lambda: gr.update(choices=["None"] + get_lora_choices()),
            outputs=lora_dropdown
        )
        
        # Define helper functions for character/persona management
        def refresh_char_selector():
            chars = db.get_all_ai_characters()
            display_names = ["➕ Create New"] + [f"{char['avatar']} {char['name']}" for char in chars]
            return gr.update(choices=display_names)
        
        def refresh_persona_selector():
            personas = db.get_all_user_personas()
            display_names = ["➕ Create New"] + [f"{p['avatar']} {p['name']}" for p in personas]
            return gr.update(choices=display_names)
        
        def open_char_editor(display_name):
            if display_name is None:
                return gr.update(visible=True), gr.update(visible=False), "", "", "", "", 0.7, 0.95, 50, "🤖"
            # Extract actual name from display format "🤖 Name"
            if display_name.startswith("➕"):
                # Return empty form for new character
                return gr.update(visible=False), gr.update(visible=True), "", "", "", "", 0.7, 0.95, 50, "🤖"
            else:
                actual_name = display_name.split(" ", 1)[1] if " " in display_name else display_name
                # Load existing character data
                ai_characters = db.get_all_ai_characters()
                for char in ai_characters:
                    if char['name'] == actual_name:
                        return (
                            gr.update(visible=False),  # hide browse
                            gr.update(visible=True),  # show edit
                            str(char['id']),  # hidden id
                            char['name'],
                            char['system_prompt'] or "",
                            char['description'] or "",
                            char['temperature'],
                            char['top_p'],
                            char['top_k'],
                            char['avatar'] or "🤖"
                        )
                # If not found, return empty
                return gr.update(visible=False), gr.update(visible=True), "", actual_name, "", "", 0.7, 0.95, 50, "🤖"
        
        def open_persona_editor(display_name):
            if display_name is None:
                return gr.update(visible=True), gr.update(visible=False), "", "", "", "", "👤"
            # Extract actual name from display format "👤 Name"
            if display_name.startswith("➕"):
                # Return empty form for new persona
                return gr.update(visible=False), gr.update(visible=True), "", "", "", "", "👤"
            else:
                actual_name = display_name.split(" ", 1)[1] if " " in display_name else display_name
                # Load existing persona data
                personas = db.get_all_user_personas()
                for persona in personas:
                    if persona['name'] == actual_name:
                        return (
                            gr.update(visible=False),  # hide browse
                            gr.update(visible=True),  # show edit
                            str(persona['id']),  # hidden id
                            persona['name'],
                            persona['description'] or "",
                            persona['background'] or "",
                            persona['avatar'] or "👤"
                        )
                # If not found, return empty
                return gr.update(visible=False), gr.update(visible=True), "", actual_name, "", "", "👤"
        
        # AI Character Management Events
        # Save (create or update) character
        save_ai_btn.click(
            save_ai_character,
            [ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar],
            [ai_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            refresh_char_selector,
            outputs=ai_char_selector
        )
        
        # Delete character
        delete_ai_btn.click(
            delete_ai_character_by_name,
            [ai_char_id_hidden, new_ai_name],
            [ai_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            lambda: ("", "", "", "", 0.7, 0.95, 50, "🤖"),
            outputs=[ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar]
        ).then(
            refresh_char_selector,
            outputs=ai_char_selector
        )
        
        # Tab navigation buttons for characters
        back_to_browse_ai_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False), None),
            outputs=[char_browse_view, char_edit_view, ai_char_selector]
        )
        
        # Auto-navigate on tile selection
        ai_char_selector.change(
            open_char_editor,
            inputs=ai_char_selector,
            outputs=[char_browse_view, char_edit_view, ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar]
        )
        
        # Refresh character selector
        refresh_ai_tiles_btn.click(refresh_char_selector, outputs=ai_char_selector)
        
        # User Persona Management Events
        # Save (create or update) persona
        save_user_btn.click(
            save_user_persona,
            [user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar],
            [user_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            refresh_persona_selector,
            outputs=user_persona_selector
        )
        
        # Delete persona
        delete_user_btn.click(
            delete_user_persona_by_name,
            [user_persona_id_hidden, new_user_name],
            [user_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            lambda: ("", "", "", "", "👤"),
            outputs=[user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar]
        ).then(
            refresh_persona_selector,
            outputs=user_persona_selector
        )
        
        # Tab navigation buttons for personas
        back_to_browse_user_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False), None),
            outputs=[persona_browse_view, persona_edit_view, user_persona_selector]
        )
        
        # Auto-navigate on tile selection
        user_persona_selector.change(
            open_persona_editor,
            inputs=user_persona_selector,
            outputs=[persona_browse_view, persona_edit_view, user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar]
        )
        
        # Refresh persona selector
        refresh_user_tiles_btn.click(refresh_persona_selector, outputs=user_persona_selector)
    
    return demo


def main():
    """Main."""
    global gen_config, cache_dir, db
    
    print("=" * 60, flush=True)
    print("KVGenius Multi-Model Chat", flush=True)
    print("=" * 60, flush=True)
    
    load_environment()
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    gen_config = config.get('generation', {})
    cache_dir = config.get('model', {}).get('cache_dir', './data/model_cache')
    
    # Initialize database
    db = ChatHistoryDB(db_path="./data/chat_history.db")
    db.init_defaults()
    
    print(f"\n✓ Config loaded", flush=True)
    print(f"✓ Cache: {cache_dir}", flush=True)
    print(f"✓ Database: data/chat_history.db", flush=True)
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"✓ PyTorch: {torch.__version__}", flush=True)
    
    # Skip pre-loading to save startup time (models load on first use)
    # default_model = "Nous-Hermes-2-Mistral-7B"
    # print(f"\n⏳ Pre-loading {default_model}...")
    # try:
    #     load_model(default_model)
    #     print(f"✓ {default_model} loaded and ready")
    # except Exception as e:
    #     print(f"⚠ Could not pre-load model: {e}")
    #     print("  (Will load on first use instead)")
    
    demo = create_ui()
    
    print("\n" + "=" * 60, flush=True)
    print("✓ Starting server on http://127.0.0.1:7860", flush=True)
    print("=" * 60, flush=True)
    print("\nPress Ctrl+C to stop\n", flush=True)
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
