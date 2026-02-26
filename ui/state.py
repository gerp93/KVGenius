"""
Global application state for KVGenius Desktop App.
This module provides shared state across all UI components.
"""
from typing import Optional
from core import get_setting


class AppState:
    """Global application state - mirrors Gradio's global variables."""
    
    def __init__(self):
        self.loaded_model: Optional[str] = None  # Currently loaded model name
        self.loaded_model_type: Optional[str] = None  # "image" or "chat"
        self.image_model_arch: Optional[str] = None  # "sd", "sdxl", "flux" - architecture type
        # Legacy compatibility
        self.image_model_loaded: Optional[str] = None
        self.chat_model_loaded: Optional[str] = None
        # Load theme from settings, default to "dark"
        self.current_theme: str = get_setting("theme", "dark")
        self.is_generating: bool = False
        self.cancel_requested: bool = False


# Global singleton instance
app_state = AppState()
