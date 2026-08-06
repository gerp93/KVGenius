"""
Global application state for KVGenius Desktop App.
This module provides shared state across all UI components.
"""
from typing import Optional
from core import get_setting
from theming import DEFAULT_THEME_ID, theme_exists


class AppState:
    """Global application state - mirrors Gradio's global variables."""
    
    def __init__(self):
        self.loaded_model: Optional[str] = None  # Currently loaded model name
        self.loaded_model_type: Optional[str] = None  # "image" or "chat"
        self.image_model_arch: Optional[str] = None  # "sd", "sdxl", "flux" - architecture type
        # Legacy compatibility
        self.image_model_loaded: Optional[str] = None
        self.chat_model_loaded: Optional[str] = None
        # Load theme from settings, default to a dark VisualAssault theme.
        # Falls back past a stale/pre-migration setting value (e.g. the old
        # flet_kvg_themes "dark" id, which VisualAssault doesn't have).
        saved_theme = get_setting("theme", DEFAULT_THEME_ID)
        self.current_theme: str = saved_theme if theme_exists(saved_theme) else DEFAULT_THEME_ID
        self.is_generating: bool = False
        self.cancel_requested: bool = False


# Global singleton instance
app_state = AppState()
