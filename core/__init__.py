# KVGenius Core Module
# UI-agnostic backend logic for image generation, chat, and model management

from .config import (
    PROJECT_ROOT,
    CACHE_DIR,
    LOG_DIR,
    CHECKPOINTS_DIR,
    LORA_DIR,
    GENERATED_IMAGES_DIR,
    CONFIG_DIR,
    load_config,
    load_image_model_presets,
    load_chat_model_presets,
    HUGGINGFACE_MODELS,
    LOCAL_CHECKPOINTS,
    PROMPT_TEMPLATES,
    CHAT_MODELS,
)

from .image_gen import (
    get_available_image_models,
    is_model_downloaded,
    load_image_model,
    unload_image_model,
    generate_image,
    get_current_model,
    get_generated_images,
    GenerationResult,
)

__all__ = [
    # Config
    "PROJECT_ROOT",
    "CACHE_DIR",
    "LOG_DIR",
    "CHECKPOINTS_DIR",
    "LORA_DIR",
    "GENERATED_IMAGES_DIR",
    "CONFIG_DIR",
    "load_config",
    "load_image_model_presets",
    "load_chat_model_presets",
    "HUGGINGFACE_MODELS",
    "LOCAL_CHECKPOINTS",
    "PROMPT_TEMPLATES",
    "CHAT_MODELS",
    # Image Generation
    "get_available_image_models",
    "is_model_downloaded",
    "load_image_model",
    "unload_image_model",
    "generate_image",
    "get_current_model",
    "get_generated_images",
    "GenerationResult",
]
