"""
Header Bar component for KVGenius Desktop App.
Contains the unified model selector and load/unload functionality.
"""
import threading
from typing import Optional, Callable

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, ElevatedButton, Dropdown,
    IconButton, Colors, Icons, MainAxisAlignment, FontWeight,
    padding, dropdown,
)

from ui.state import app_state
from ui.common import get_gpu_info
from core import (
    get_available_image_models,
    get_available_chat_models,
    is_model_downloaded,
    is_chat_model_downloaded,
    load_image_model,
    unload_image_model,
    load_chat_model,
    unload_chat_model,
)


def get_unified_model_choices():
    """Get all model choices (image + chat) with status indicators."""
    options = []
    
    # Image Models Section
    image_models = get_available_image_models()
    if image_models:
        options.append(dropdown.Option("__image_header__", "───── 🎨 IMAGE MODELS ─────", disabled=True))
        
        for name, info in image_models.items():
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "image")
            is_local = info.get("source") == "local"
            is_downloaded = is_local or is_model_downloaded(info.get("id", ""))
            
            if is_loaded:
                status = "🟢"
            elif is_downloaded:
                status = "🟡"
            else:
                status = "🔴"
            
            model_type = info.get("type", "sd").upper()
            vram = info.get("vram", "")
            
            options.append(dropdown.Option(
                f"image:{name}",
                f"  {status} {name} [{model_type}] {vram}",
            ))
    
    # Chat Models Section
    chat_models = get_available_chat_models()
    if chat_models:
        options.append(dropdown.Option("__chat_header__", "───── 💬 CHAT/LLM MODELS ─────", disabled=True))
        
        for name, info in chat_models.items():
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "chat")
            is_downloaded = is_chat_model_downloaded(info.get("id", ""))
            
            if is_loaded:
                status = "🟢"
            elif is_downloaded:
                status = "🟡"
            else:
                status = "🔴"
            
            vram = info.get("vram", "~14 GB")
            
            options.append(dropdown.Option(
                f"chat:{name}",
                f"  {status} {name} {vram}",
            ))
    
    return options


def get_model_info(model_key: str) -> str:
    """Get detailed info for a model."""
    if ":" in model_key:
        model_type, model_name = model_key.split(":", 1)
    else:
        model_type, model_name = "image", model_key
    
    if model_type == "image":
        models = get_available_image_models()
        if model_name not in models:
            return f"Unknown model: {model_name}"
        
        info = models[model_name]
        is_loaded = (app_state.loaded_model == model_name and app_state.loaded_model_type == "image")
        is_downloaded = info.get("source") == "local" or is_model_downloaded(info.get("id", ""))
        
        status = "🟢 LOADED" if is_loaded else ("🟡 Downloaded" if is_downloaded else "🔴 Not Downloaded")
        
        lines = [
            f"🎨 {model_name} | {status}",
            f"Type: {info.get('type', 'sd').upper()} | VRAM: {info.get('vram', 'Unknown')}",
        ]
        if info.get("description"):
            lines.append(info['description'])
        return " | ".join(lines)
    
    else:
        models = get_available_chat_models()
        if model_name not in models:
            return f"Unknown model: {model_name}"
        
        info = models[model_name]
        is_loaded = (app_state.loaded_model == model_name and app_state.loaded_model_type == "chat")
        is_downloaded = is_chat_model_downloaded(info.get("id", ""))
        
        status = "🟢 LOADED" if is_loaded else ("🟡 Downloaded" if is_downloaded else "🔴 Not Downloaded")
        
        lines = [
            f"💬 {model_name} | {status}",
            f"VRAM: {info.get('vram', '~14 GB')}",
        ]
        if info.get("description"):
            lines.append(info['description'])
        return " | ".join(lines)


class HeaderBar:
    """Global header with unified model selector."""
    
    def __init__(self, page: Page, on_model_loaded: Callable = None, theme_dropdown: Dropdown = None):
        self.page = page
        self.on_model_loaded = on_model_loaded
        self.theme_dropdown = theme_dropdown
        self._build_ui()
    
    def _build_ui(self):
        options = get_unified_model_choices()
        
        first_value = None
        for opt in options:
            if opt.key not in ("__image_header__", "__chat_header__"):
                first_value = opt.key
                break
        
        self.model_dropdown = Dropdown(
            label="🤖 Model",
            width=450,
            options=options,
            value=first_value,
            on_change=self._on_model_change,
        )
        
        self.gpu_info = Text(get_gpu_info(), size=11, color=Colors.GREY_500)
        
        self.load_btn = ElevatedButton(
            "📥 Load",
            on_click=self._load_model,
            bgcolor=Colors.BLUE_700,
            color=Colors.WHITE,
        )
        self.unload_btn = ElevatedButton(
            "🗑️ Unload",
            on_click=self._unload_model,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            visible=False,
        )
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            on_click=self._refresh_models,
            tooltip="Refresh model list",
        )
        
        self.status_text = Text(
            "💡 Select a model to see details, then click Load to start",
            size=13,
            color=Colors.GREY_400,
        )
    
    def _on_model_change(self, e):
        """Update status when model selection changes."""
        model_key = self.model_dropdown.value
        if model_key:
            info = get_model_info(model_key)
            self.status_text.value = info
            self._update_button_visibility()
            self.page.update()
    
    def _refresh_models(self, e):
        """Refresh the model dropdown."""
        current_value = self.model_dropdown.value
        options = get_unified_model_choices()
        self.model_dropdown.options = options
        self.model_dropdown.value = current_value
        self._update_button_visibility()
        self.page.update()
    
    def _update_button_visibility(self):
        """Toggle Load/Unload button visibility."""
        selected_key = self.model_dropdown.value
        
        if selected_key and ":" in selected_key:
            sel_type, sel_name = selected_key.split(":", 1)
        else:
            sel_type, sel_name = None, None
        
        is_selected_loaded = (
            app_state.loaded_model == sel_name and 
            app_state.loaded_model_type == sel_type
        )
        
        if is_selected_loaded:
            self.load_btn.visible = False
            self.unload_btn.visible = True
        else:
            self.load_btn.visible = True
            self.unload_btn.visible = False
    
    def _load_model(self, e):
        """Load the selected model."""
        model_key = self.model_dropdown.value
        if not model_key:
            return
        
        if ":" in model_key:
            model_type, model_name = model_key.split(":", 1)
        else:
            model_type, model_name = "image", model_key
        
        self.load_btn.disabled = True
        self.status_text.value = f"⏳ Loading {model_name}..."
        self.page.update()
        
        def do_load():
            def progress_cb(pct, msg):
                self.status_text.value = f"⏳ [{pct*100:.0f}%] {msg}"
                try:
                    self.page.update()
                except:
                    pass
            
            # Unload existing model first
            if app_state.loaded_model:
                if app_state.loaded_model_type == "image":
                    unload_image_model()
                else:
                    unload_chat_model()
                app_state.loaded_model = None
                app_state.loaded_model_type = None
                app_state.image_model_loaded = None
                app_state.chat_model_loaded = None
                app_state.image_model_arch = None
            
            # Load new model
            if model_type == "image":
                success, message = load_image_model(model_name, progress_callback=progress_cb)
                if success:
                    app_state.loaded_model = model_name
                    app_state.loaded_model_type = "image"
                    app_state.image_model_loaded = model_name
                    models = get_available_image_models()
                    if model_name in models:
                        app_state.image_model_arch = models[model_name].get("type", "sd")
            else:
                success, message = load_chat_model(model_name, progress_callback=progress_cb)
                if success:
                    app_state.loaded_model = model_name
                    app_state.loaded_model_type = "chat"
                    app_state.chat_model_loaded = model_name
            
            if success:
                emoji = "🎨" if model_type == "image" else "💬"
                self.status_text.value = f"🟢 {emoji} {model_name} loaded and ready!"
            else:
                self.status_text.value = f"❌ Failed: {message}"
            
            self.load_btn.disabled = False
            self._update_button_visibility()
            self._refresh_models(None)
            
            if self.on_model_loaded:
                self.on_model_loaded(success, model_name if success else None)
            
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=do_load, daemon=True).start()
    
    def _unload_model(self, e):
        """Unload the current model."""
        model_name = app_state.loaded_model
        model_type = app_state.loaded_model_type
        if not model_name:
            return
        
        self.status_text.value = f"🗑️ Unloading {model_name}..."
        self.page.update()
        
        if model_type == "image":
            unload_image_model()
        else:
            unload_chat_model()
        
        app_state.loaded_model = None
        app_state.loaded_model_type = None
        app_state.image_model_loaded = None
        app_state.chat_model_loaded = None
        app_state.image_model_arch = None
        
        self.status_text.value = "💡 Model unloaded. Select a model to load."
        self._update_button_visibility()
        self._refresh_models(None)
        
        if self.on_model_loaded:
            self.on_model_loaded(False, None)
        
        self.page.update()
    
    def build(self) -> Container:
        """Build the header bar."""
        model_row_items = [
            self.model_dropdown,
            self.load_btn,
            self.unload_btn,
            self.refresh_btn,
        ]
        
        if self.theme_dropdown:
            model_row_items.insert(0, self.theme_dropdown)
            model_row_items.insert(1, Container(width=20))
        
        return Container(
            content=Column([
                Row([
                    Column([
                        Text("🤖 KVGenius AI Studio", size=22, weight=FontWeight.BOLD),
                        self.gpu_info,
                    ], spacing=2),
                    Container(expand=True),
                    Row(model_row_items, spacing=8),
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Container(
                    content=self.status_text,
                    padding=padding.only(top=8),
                ),
            ]),
            padding=padding.all(15),
        )
