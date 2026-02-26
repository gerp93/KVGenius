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


def save_config(config_name: str, config: Dict[str, Any]) -> bool:
    """Save a YAML config file to the config directory."""
    config_path = CONFIG_DIR / config_name
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False


def load_user_settings() -> Dict[str, Any]:
    """Load user settings from settings.yaml."""
    return load_config("settings.yaml")


def save_user_settings(settings: Dict[str, Any]) -> bool:
    """Save user settings to settings.yaml."""
    return save_config("settings.yaml", settings)


def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value."""
    settings = load_user_settings()
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> bool:
    """Set a single setting value."""
    settings = load_user_settings()
    settings[key] = value
    return save_user_settings(settings)


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
