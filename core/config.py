"""
Core configuration for KVGenius.
Shared paths and settings used by both Gradio and Flet UIs.
"""
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# Project root directory. In a packaged build this is the app's install
# directory (e.g. "C:\Program Files (x86)\KVGenius\app") - read-only to a
# non-admin user, so it's only safe to use for bundled, read-only resources
# (config/ defaults and templates), never as a place to create or write files.
PROJECT_ROOT = Path(__file__).parent.parent


def _get_user_data_dir() -> Path:
    """Per-user, always-writable directory for KVGenius's own data, separate
    from PROJECT_ROOT since that may be a read-only install location."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "KVGenius"


USER_DATA_DIR = _get_user_data_dir()

# Standard directories (user-writable - never under PROJECT_ROOT)
CACHE_DIR = USER_DATA_DIR / "data" / "model_cache"
LOG_DIR = USER_DATA_DIR / "logs"
CHECKPOINTS_DIR = USER_DATA_DIR / "data" / "checkpoints"
LORA_DIR = USER_DATA_DIR / "data" / "lora_models"
GENERATED_IMAGES_DIR = USER_DATA_DIR / "data" / "generated_images"
USER_CONFIG_DIR = USER_DATA_DIR / "config"

# Bundled application resources (defaults, examples, ComfyUI workflow
# templates) - read-only, shipped alongside the app itself.
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure user-writable directories exist
for d in [CACHE_DIR, LOG_DIR, CHECKPOINTS_DIR, LORA_DIR, GENERATED_IMAGES_DIR, USER_CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML config file: a saved user override if one exists, else the
    app's bundled default, else its .example.yaml template."""
    user_path = USER_CONFIG_DIR / config_name
    bundled_path = CONFIG_DIR / config_name
    example_path = CONFIG_DIR / config_name.replace('.yaml', '.example.yaml')

    if user_path.exists():
        path_to_use = user_path
    elif bundled_path.exists():
        path_to_use = bundled_path
    else:
        path_to_use = example_path

    if not path_to_use.exists():
        return {}

    with open(path_to_use, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_config(config_name: str, config: Dict[str, Any]) -> bool:
    """Save a YAML config file as a user override, under the writable
    USER_CONFIG_DIR (never into PROJECT_ROOT, which a packaged install may
    not have permission to write to)."""
    config_path = USER_CONFIG_DIR / config_name
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
