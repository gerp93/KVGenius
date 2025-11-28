"""
Image generation presets and templates configuration.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

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

# Default image models (fallback if config file not found)
DEFAULT_IMAGE_MODELS = {
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


def load_image_model_presets(config_dir=None):
    """Load image model presets from config file.
    
    Args:
        config_dir: Path to config directory. If None, uses default location.
    
    Returns:
        Tuple of (models_dict, prompt_templates_dict)
    """
    if config_dir is None:
        # Default to config/ in project root
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    
    presets_path = os.path.join(config_dir, "image_model_presets.yaml")
    
    try:
        with open(presets_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('models', {}), config.get('prompt_templates', {})
    except FileNotFoundError:
        logger.warning(f"Image model presets not found at {presets_path}, using defaults")
        return {}, {}
    except Exception as e:
        logger.warning(f"Could not load image model presets: {e}")
        return {}, {}


def build_available_models(presets=None):
    """Build AVAILABLE_IMAGE_MODELS dict from presets.
    
    Args:
        presets: Dict of model presets. If None, loads from config file.
    
    Returns:
        Dict of available image models with full config
    """
    if presets is None:
        presets, _ = load_image_model_presets()
    
    if not presets:
        return DEFAULT_IMAGE_MODELS.copy()
    
    available = {}
    for model_key, preset in presets.items():
        available[model_key] = {
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
            "tips": preset.get("tips", "")
        }
    
    return available
