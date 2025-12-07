"""
Core configuration for KVGenius.
Shared paths and settings used by both Gradio and Flet UIs.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Standard directories
CACHE_DIR = PROJECT_ROOT / "data" / "model_cache"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "data" / "checkpoints"
LORA_DIR = PROJECT_ROOT / "data" / "lora_models"
GENERATED_IMAGES_DIR = PROJECT_ROOT / "data" / "generated_images"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for d in [CACHE_DIR, LOG_DIR, CHECKPOINTS_DIR, LORA_DIR, GENERATED_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML config file from the config directory."""
    config_path = CONFIG_DIR / config_name
    example_path = CONFIG_DIR / config_name.replace('.yaml', '.example.yaml')
    
    # Use user config if exists, otherwise example
    path_to_use = config_path if config_path.exists() else example_path
    
    if not path_to_use.exists():
        return {}
    
    with open(path_to_use, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_image_model_presets() -> Tuple[Dict, Dict, Dict]:
    """Load image model presets from config file.
    
    Returns:
        Tuple of (huggingface_models, local_checkpoints, prompt_templates)
    """
    config = load_config("image_model_presets.yaml")
    
    huggingface = config.get('huggingface', {}) or config.get('models', {})
    local = config.get('local', {}) or config.get('checkpoints', {})
    prompt_templates = config.get('prompt_templates', {})
    
    return huggingface, local, prompt_templates


def load_chat_model_presets() -> Dict[str, Any]:
    """Load chat model presets from config file."""
    config = load_config("chat_model_presets.yaml")
    return config.get('models', {})


# Load presets at module import
HUGGINGFACE_MODELS, LOCAL_CHECKPOINTS, PROMPT_TEMPLATES = load_image_model_presets()
CHAT_MODELS = load_chat_model_presets()
