#!/usr/bin/env python3
"""
KVGenius Desktop App - Modular Entry Point
==========================================
AI Image Generator & Chat application using Flet UI framework.

This is the modular version that imports tab components from the ui/ package.
"""

# Fix DLL paths before importing torch
import fix_dll_paths

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flet imports
import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Tabs, Tab, Dropdown, dropdown,
    Colors, FontWeight, padding, border_radius,
)

# Import application state
from ui.state import app_state

# Import tab components
from ui.tabs import (
    ImageGenTab,
    GalleryTab,
    PromptLibraryTab,
    ChatTab,
    SettingsTab,
    ModelHubTab,
    CardGeneratorTab,
)

# Import header component
from ui.components.header import HeaderBar

# Import core configuration
from core.config import get_setting, set_setting

# Try to import themes
THEMES_AVAILABLE = False
get_theme = None
get_theme_list = None
try:
    from kvg_themes import get_theme, get_theme_list
    THEMES_AVAILABLE = True
    logger.info("KVG Themes available")
except ImportError:
    logger.warning("kvg_themes not found, using default theme")


def main(page: Page):
    """Main application entry point."""
    page.title = "KVGenius - AI Image Generator"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window.width = 1200
    page.window.height = 900
    
    # Apply theme from KVG Themes if available
    if THEMES_AVAILABLE and get_theme:
        theme = get_theme(app_state.current_theme)
        if theme:
            page.theme = theme
            logger.info(f"Applied theme: {app_state.current_theme}")
    
    # Theme change handler
    def on_theme_change(e):
        if THEMES_AVAILABLE and get_theme and e.control.value:
            theme_id = e.control.value
            theme = get_theme(theme_id)
            if theme:
                page.theme = theme
                app_state.current_theme = theme_id
                # Save to settings file for persistence
                set_setting("theme", theme_id)
                page.update()
                logger.info(f"Theme changed to: {theme_id}")
    
    # Create theme selector dropdown
    theme_dropdown = None
    if THEMES_AVAILABLE and get_theme_list:
        themes = get_theme_list()
        theme_dropdown = Dropdown(
            label="🎨 Theme",
            width=220,
            options=[dropdown.Option(key=tid, text=name) for tid, name in themes],
            value=app_state.current_theme,
            on_change=on_theme_change,
        )
    
    # Create prompt library first (needed for image gen callback)
    # We'll set the on_switch_tab callback later after tabs is created
    prompt_library_tab = PromptLibraryTab(page)
    
    # Callback when user wants to save prompt from image gen
    def on_save_prompt(prompt: str, negative: str, settings: dict = None):
        prompt_library_tab.add_prompt_from_generator(prompt, negative, settings)
    
    # Create tabs
    image_gen_tab = ImageGenTab(page, on_save_prompt=on_save_prompt)
    gallery_tab = GalleryTab(page)
    chat_tab = ChatTab(page)
    
    # Callback to set prompt in image gen from library
    def on_use_prompt(prompt: str, negative: str):
        image_gen_tab.prompt_field.value = prompt
        if negative:
            image_gen_tab.negative_field.value = negative
        page.update()
    
    # Connect prompt library callback
    prompt_library_tab.on_use_prompt = on_use_prompt
    
    # Callback when model loads/unloads
    def on_model_loaded(success: bool, model_key: Optional[str]):
        # Update image gen tab
        image_gen_tab.update_model_state()
        # Update chat tab
        chat_tab.update_model_state()
    
    # Settings tab
    settings_tab = SettingsTab(page)
    
    # Model Hub tab (combines Models, LoRAs, Datasets, Training)
    model_hub_tab = ModelHubTab(page)
    
    # Card Generator tab
    card_generator_tab = CardGeneratorTab(page)
    
    # Create header
    header = HeaderBar(page, on_model_loaded=on_model_loaded, theme_dropdown=theme_dropdown)
    
    # Build tabs
    tabs = Tabs(
        selected_index=0,
        animation_duration=200,
        tabs=[
            Tab(
                text="🖌️ Generate",
                content=image_gen_tab.build(),
            ),
            Tab(
                text="💬 Chat",
                content=chat_tab.build(),
            ),
            Tab(
                text="🃏 Cards",
                content=card_generator_tab.build(),
            ),
            Tab(
                text="📚 Prompts",
                content=prompt_library_tab.build(),
            ),
            Tab(
                text="🖼️ Gallery",
                content=gallery_tab.build(),
            ),
            Tab(
                text="🧠 Model Hub",
                content=model_hub_tab.build(),
            ),
            Tab(
                text="⚙️ Settings",
                content=settings_tab.build(),
            ),
        ],
        expand=True,
    )
    
    # Set the callback for switching to generate tab after tabs is created
    def on_switch_to_tab(tab_index: int):
        """Callback to switch to a specific tab."""
        tabs.selected_index = tab_index
        page.update()
    
    prompt_library_tab.on_switch_tab = on_switch_to_tab
    
    page.add(
        Column([
            header.build(),
            tabs,
        ], expand=True, spacing=0)
    )


if __name__ == "__main__":
    ft.app(target=main)
