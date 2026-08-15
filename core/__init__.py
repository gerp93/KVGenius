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
    save_config,
    load_user_settings,
    save_user_settings,
    get_setting,
    set_setting,
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
    _is_model_fully_downloaded,
    load_image_model,
    unload_image_model,
    generate_image,
    get_current_model,
    get_generated_images,
    get_available_loras,
    unload_lora,
    download_model,
    GenerationResult,
)

from .chat_gen import (
    get_available_chat_models,
    is_chat_model_downloaded,
    _is_chat_model_fully_downloaded,
    get_current_chat_model,
    load_chat_model,
    unload_chat_model,
    generate_chat_response,
    clear_conversation,
    get_conversation_history,
    download_chat_model,
    parse_message_with_actions,
    build_message_with_directions,
    _last_debug_info,
)

from .prompt_builder import (
    PromptBuilder,
    load_prompt_templates,
    save_prompt_templates,
)

from .semantic_index import (
    SemanticRetriever,
    get_retriever,
    retrieve_memories,
    RetrievalResult,
    ScoredChunk,
)

from .db_location import (
    db_location,
    get_effective_db_path,
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
    "save_config",
    "load_user_settings",
    "save_user_settings",
    "get_setting",
    "set_setting",
    "load_image_model_presets",
    "load_chat_model_presets",
    "HUGGINGFACE_MODELS",
    "LOCAL_CHECKPOINTS",
    "PROMPT_TEMPLATES",
    "CHAT_MODELS",
    # Image Generation
    "get_available_image_models",
    "is_model_downloaded",
    "_is_model_fully_downloaded",
    "load_image_model",
    "unload_image_model",
    "generate_image",
    "get_current_model",
    "get_generated_images",
    "get_available_loras",
    "unload_lora",
    "download_model",
    "GenerationResult",
    # Chat Generation
    "get_available_chat_models",
    "is_chat_model_downloaded",
    "_is_chat_model_fully_downloaded",
    "get_current_chat_model",
    "load_chat_model",
    "unload_chat_model",
    "generate_chat_response",
    "clear_conversation",
    "get_conversation_history",
    "download_chat_model",
    "parse_message_with_actions",
    "build_message_with_directions",
    "_last_debug_info",
    # Prompt Builder
    "PromptBuilder",
    "load_prompt_templates",
    "save_prompt_templates",
    # Semantic Retrieval
    "SemanticRetriever",
    "get_retriever",
    "retrieve_memories",
    "RetrievalResult",
    "ScoredChunk",
    # DB location
    "db_location",
    "get_effective_db_path",
]
