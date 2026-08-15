"""
KVGenius Desktop Application
Built with Flet for local GPU-powered AI generation.

Design mirrors the proven Gradio UI:
- Global header with model selectors
- Load/Unload buttons toggle visibility
- Status bar with model info
- Exclusive model loading (image unloads chat, etc.)

Requirements:
- Run with kvgen conda environment
- RTX 5070 Ti or compatible GPU
"""
import os
import sys
import logging
import threading
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Set

# Fix DLL paths before importing torch
sys.path.insert(0, str(Path(__file__).parent))
import fix_dll_paths

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Card, Tabs, Tab,
    TextField, ElevatedButton, ProgressBar, Image, Dropdown,
    Slider, IconButton, ListView, Colors, Icons,
    MainAxisAlignment, CrossAxisAlignment, ScrollMode,
    dropdown, SnackBar, AlertDialog, TextButton, FontWeight,
    padding, border_radius, border, GridView, Checkbox, Stack, alignment,
    ControlState, ButtonStyle, Divider,
)

# Import VisualAssault themes (theming.py wraps visual_assault_flet)
from theming import get_theme, get_theme_list, get_theme_background, theme_exists
THEMES_AVAILABLE = True

# Import core engine
from core import (
    get_available_image_models,
    is_model_downloaded,
    _is_model_fully_downloaded,
    load_image_model,
    unload_image_model,
    generate_image,
    get_current_model,
    get_generated_images,
    get_available_loras,
    download_model,
    get_setting,
    set_setting,
    GenerationResult,
    HUGGINGFACE_MODELS,
    # Chat
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
)

# Import database for character/persona management
from src.database import ChatHistoryDB

# Import CardGeneratorTab from ui.tabs
from ui.tabs.card_generator import CardGeneratorTab

# Self-update (kvg_updater bundle mode, see gerp93/KVG_Standards)
from updater import CURRENT_VERSION, check_for_update, check_and_apply_update

# Database location (kvg_dblocation, see gerp93/KVG_Standards' db-location-versioning.md)
from core import db_location as chat_db_location


def get_gpu_info():
    """Get GPU name and compute capability for display."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            return f"{gpu_name} (sm_{major}{minor})"
        else:
            return "CPU (No GPU detected)"
    except:
        return "GPU detection unavailable"


def _restart_app():
    """Re-exec the current process. Used after a database location change
    (an already-open sqlite3 connection can't be pointed at a new file) —
    see gerp93/KVG_Standards' db-location-versioning.md. Also works for a
    PyInstaller/Flet-built executable, where sys.executable is the built
    binary itself rather than a `python` interpreter."""
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Set up logging with flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    force=True,
)

# Force unbuffered output
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

def debug_print(msg):
    """Print with immediate flush."""
    sys.stdout.write(f"[DEBUG] {msg}\n")
    sys.stdout.flush()

logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL STATE - Import from ui.state for shared state across tabs
# =============================================================================
from ui.state import app_state


# =============================================================================
# HELPER FUNCTIONS - Match Gradio's logic
# =============================================================================
def get_unified_model_choices():
    """Get all model choices (image + chat) with status indicators and sections."""
    options = []
    
    # Image Models Section
    image_models = get_available_image_models()
    if image_models:
        # Add section header (disabled option)
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
        # Add section header (disabled option)
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
    
    return choices


def get_image_model_choices():
    """Get image model choices with status indicators (matches Gradio)."""
    models = get_available_image_models()
    choices = []
    
    for name, info in models.items():
        # Determine status
        is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "image")
        is_local = info.get("source") == "local"
        is_downloaded = is_local or is_model_downloaded(info.get("id", ""))
        
        # Status indicator (matches Gradio: 🟢 Loaded, 🟡 Downloaded, 🔴 Not Downloaded)
        if is_loaded:
            status = "🟢"
        elif is_downloaded:
            status = "🟡"
        else:
            status = "🔴"
        
        model_type = info.get("type", "sd").upper()
        vram = info.get("vram", "")
        
        choices.append({
            "key": name,
            "text": f"{status} {name} [{model_type}] {vram}",
            "downloaded": is_downloaded,
            "loaded": is_loaded,
        })
    
    return choices


def get_model_info(model_key: str) -> str:
    """Get detailed info for a model (image or chat)."""
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
    
    else:  # chat
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


def get_image_model_info(model_name: str) -> str:
    """Get detailed info for a model (matches Gradio's get_image_model_info)."""
    return get_model_info(f"image:{model_name}")


# =============================================================================
# HEADER BAR - Global Model Selectors (matches Gradio header)
# =============================================================================
class HeaderBar:
    """Global header with unified model selector - only one model loaded at a time."""
    
    def __init__(self, page: Page, on_model_loaded: callable = None, theme_dropdown: Dropdown = None):
        self.page = page
        self.on_model_loaded = on_model_loaded
        self.theme_dropdown = theme_dropdown
        self._build_ui()
    
    def _build_ui(self):
        # Unified Model Dropdown (Image + Chat models with sections)
        options = get_unified_model_choices()
        
        # Find first selectable option (skip section headers)
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
        
        # GPU info text
        self.gpu_info = Text(get_gpu_info(), size=11, color=Colors.GREY_500)
        
        # Load/Unload buttons (toggle visibility)
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
            visible=False,  # Hidden until model loaded
        )
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            on_click=self._refresh_models,
            tooltip="Refresh model list",
        )
        
        # Status bar
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
        """Refresh the model dropdown and preserve selection."""
        current_value = self.model_dropdown.value
        options = get_unified_model_choices()
        self.model_dropdown.options = options
        # Restore the selection (it should match one of the new options by key)
        self.model_dropdown.value = current_value
        self._update_button_visibility()
        self.page.update()
    
    def _update_button_visibility(self):
        """Toggle Load/Unload button visibility based on state and selection."""
        selected_key = self.model_dropdown.value
        
        # Parse selected model
        if selected_key and ":" in selected_key:
            sel_type, sel_name = selected_key.split(":", 1)
        else:
            sel_type, sel_name = None, None
        
        # Determine if the selected model is the currently loaded one
        is_selected_loaded = (
            app_state.loaded_model == sel_name and 
            app_state.loaded_model_type == sel_type
        )
        
        if is_selected_loaded:
            # Selected model is loaded - show Unload only
            self.load_btn.visible = False
            self.unload_btn.visible = True
        else:
            # Different model selected or nothing loaded - show Load
            self.load_btn.visible = True
            self.unload_btn.visible = False
    
    def _load_model(self, e):
        """Load the selected model (image or chat)."""
        model_key = self.model_dropdown.value
        if not model_key:
            return
        
        # Parse model type and name
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
            
            # Unload any existing model first
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
            
            # Load the new model
            if model_type == "image":
                success, message = load_image_model(model_name, progress_callback=progress_cb)
                if success:
                    app_state.loaded_model = model_name
                    app_state.loaded_model_type = "image"
                    app_state.image_model_loaded = model_name
                    # Store model architecture type for LoRA filtering
                    models = get_available_image_models()
                    if model_name in models:
                        app_state.image_model_arch = models[model_name].get("type", "sd")
                    else:
                        app_state.image_model_arch = "sd"
            else:  # chat
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
            
            # Notify tab that model is loaded
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
        # Build the model selector row
        model_row_items = [
            self.model_dropdown,
            self.load_btn,
            self.unload_btn,
            self.refresh_btn,
        ]
        
        # Add theme dropdown if available
        if self.theme_dropdown:
            model_row_items.insert(0, self.theme_dropdown)
            model_row_items.insert(1, Container(width=20))  # Spacer
        
        return Container(
            content=Column([
                # Top row: Title + Model selectors + Theme
                Row([
                    # Title + GPU Info
                    Column([
                        Text("🤖 KVGenius AI Studio", size=22, weight=FontWeight.BOLD),
                        self.gpu_info,
                    ], spacing=2),
                    
                    # Spacer
                    Container(expand=True),
                    
                    # Theme + Unified Model selector with Load/Unload buttons
                    Row(model_row_items, spacing=8),
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                
                # Status bar
                Container(
                    content=self.status_text,
                    padding=padding.only(top=8),
                ),
            ]),
            padding=padding.all(15),
        )


# =============================================================================
# IMAGE GENERATION TAB - Dynamic layout based on aspect ratio
# =============================================================================
class ImageGenTab:
    """Image Generation Tab with dynamic image placement."""
    
    # Layout modes
    LAYOUT_AUTO = "Auto (by aspect ratio)"
    LAYOUT_RIGHT = "Image on Right"
    LAYOUT_BOTTOM = "Image on Bottom"
    
    def __init__(self, page: Page, on_save_prompt: Optional[Callable] = None):
        self.page = page
        self.on_save_prompt = on_save_prompt  # Callback to save prompt to library
        self.current_result: Optional[GenerationResult] = None
        self.current_layout = self.LAYOUT_AUTO
        self._build_ui()
    
    def set_generate_enabled(self, enabled: bool):
        """Enable/disable generate button based on model state."""
        self.generate_btn.disabled = not enabled
        self.no_model_banner.visible = not enabled
        self.page.update()
    
    def update_model_state(self):
        """Update UI based on current model state."""
        has_model = app_state.loaded_model_type == "image"
        self.generate_btn.disabled = not has_model
        self.no_model_banner.visible = not has_model
        
        # Refresh LoRA dropdown filtered by model type
        self._refresh_lora_dropdown()
        
        try:
            self.page.update()
        except:
            pass
    
    def _refresh_lora_dropdown(self):
        """Refresh LoRA dropdown filtered by loaded model type."""
        model_type = app_state.image_model_arch  # "sd", "sdxl", "flux", or None
        
        # Debug logging
        print(f"[LoRA Filter] Refreshing LoRAs for model type: {model_type}")
        
        # Get LoRAs filtered by model type (None = show all)
        available_loras = get_available_loras(model_type)
        print(f"[LoRA Filter] Found {len(available_loras)} LoRAs for type '{model_type}'")
        for lora in available_loras:
            print(f"  - {lora['name']} (type: {lora.get('model_type', 'unknown')}, base: {lora.get('base_model', 'none')})")
        
        lora_options = [dropdown.Option("None")]
        for lora in available_loras:
            lora_options.append(dropdown.Option(lora["name"]))
        
        # Update dropdown
        current_value = self.lora_dropdown.value
        self.lora_dropdown.options = lora_options
        
        # Keep current selection if still valid, otherwise reset to None
        valid_names = [opt.key for opt in lora_options]
        if current_value not in valid_names:
            self.lora_dropdown.value = "None"
    
    def _get_optimal_layout(self) -> str:
        """Determine optimal layout based on aspect ratio."""
        if self.current_layout != self.LAYOUT_AUTO:
            return self.current_layout
        
        # Get current dimensions
        width = int(self.width_dropdown.value)
        height = int(self.height_dropdown.value)
        
        # Portrait or square -> image on right (maximize vertical space)
        # Landscape -> image on bottom (maximize horizontal space)
        if height >= width:  # Portrait or square
            return self.LAYOUT_RIGHT
        else:  # Landscape
            return self.LAYOUT_BOTTOM
    
    def _on_layout_change(self, e):
        """Handle layout mode change."""
        self.current_layout = e.control.value
        self._rebuild_layout()
    
    def _on_dimension_change(self, e):
        """Handle width/height change - rebuild layout if auto."""
        if self.current_layout == self.LAYOUT_AUTO:
            self._rebuild_layout()
    
    def _rebuild_layout(self):
        """Rebuild the main content based on current layout."""
        layout = self._get_optimal_layout()
        
        # Update the main container's content
        if layout == self.LAYOUT_RIGHT:
            self.main_container.content = self._build_side_layout()
        else:
            self.main_container.content = self._build_bottom_layout()
        
        self.page.update()
    
    def _build_ui(self):
        """Build UI components."""
        
        # === LAYOUT SELECTOR ===
        self.layout_dropdown = Dropdown(
            label="Layout",
            value=self.LAYOUT_AUTO,
            options=[
                dropdown.Option(self.LAYOUT_AUTO),
                dropdown.Option(self.LAYOUT_RIGHT),
                dropdown.Option(self.LAYOUT_BOTTOM),
            ],
            on_change=self._on_layout_change,
            width=180,
        )
        
        # === SETTINGS PANEL ===
        # Load available LoRAs
        lora_options = [dropdown.Option("None")]
        for lora in get_available_loras():
            lora_options.append(dropdown.Option(lora["name"]))
        
        self.lora_dropdown = Dropdown(
            label="LoRA",
            options=lora_options,
            value="None",
            width=160,
        )
        
        # Create sliders with visible labels
        self.lora_strength_value_text = Text("", size=11)
        self.lora_strength = Slider(
            min=0.0, max=1.5, divisions=30, value=0.8,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.lora_strength_value_text, "LoRA {:.2f}", self.lora_strength.value),
        )
        
        self.steps_value_text = Text("", size=11)
        self.steps_slider = Slider(
            min=1, max=50, divisions=49, value=4,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.steps_value_text, "{} steps", int(self.steps_slider.value)),
        )
        
        self.guidance_value_text = Text("", size=11)
        self.guidance_slider = Slider(
            min=0, max=20, divisions=40, value=0,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.guidance_value_text, "CFG {:.1f}", self.guidance_slider.value),
        )
        
        self.width_dropdown = Dropdown(
            label="Width",
            expand=True,
            options=[
                dropdown.Option("512", "512"),
                dropdown.Option("576", "576"),
                dropdown.Option("640", "640"),
                dropdown.Option("704", "704"),
                dropdown.Option("768", "768"),
                dropdown.Option("832", "832"),
                dropdown.Option("896", "896"),
                dropdown.Option("960", "960"),
                dropdown.Option("1024", "1024"),
            ],
            value="512",
            on_change=self._on_dimension_change,
        )
        
        self.height_dropdown = Dropdown(
            label="Height",
            expand=True,
            options=[
                dropdown.Option("512", "512"),
                dropdown.Option("576", "576"),
                dropdown.Option("640", "640"),
                dropdown.Option("704", "704"),
                dropdown.Option("768", "768"),
                dropdown.Option("832", "832"),
                dropdown.Option("896", "896"),
                dropdown.Option("960", "960"),
                dropdown.Option("1024", "1024"),
            ],
            value="768",
            on_change=self._on_dimension_change,
        )
        self.seed_field = TextField(
            label="Seed (-1 = random)",
            value="-1",
            width=160,
            height=50,
        )
        
        # === PROMPT FIELDS ===
        self.prompt_field = TextField(
            label="✨ Prompt",
            multiline=True,
            min_lines=8,
            max_lines=20,
            hint_text="A majestic dragon flying over a crystal castle at sunset...",
            text_size=14,
            on_change=self._on_prompt_change,
        )
        
        self.negative_field = TextField(
            label="🚫 Negative Prompt",
            multiline=True,
            min_lines=4,
            max_lines=10,
            value="blurry, bad quality, distorted, ugly, deformed",
            text_size=14,
        )
        
        # === BUTTONS ===
        self.generate_btn = ElevatedButton(
            "🎨 Generate",
            on_click=self._generate,
            disabled=True,
            height=45,
            style=ButtonStyle(
                bgcolor={
                    ControlState.DEFAULT: Colors.BLUE_700,
                    ControlState.DISABLED: Colors.GREY_700,
                },
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
                overlay_color=Colors.TRANSPARENT,
            ),
        )
        self.stop_btn = ElevatedButton(
            "⏹ Stop",
            on_click=self._stop_generation,
            disabled=True,
            height=45,
            style=ButtonStyle(
                bgcolor={
                    ControlState.DEFAULT: Colors.RED_700,
                    ControlState.DISABLED: Colors.GREY_700,
                },
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
                overlay_color=Colors.TRANSPARENT,
            ),
        )
        
        self.save_prompt_btn = IconButton(
            icon=Icons.BOOKMARK_ADD,
            tooltip="Save prompt to library",
            on_click=self._save_to_library,
            disabled=True,
        )
        
        # === PROGRESS ===
        self.progress_bar = ProgressBar(value=0, width=300)
        self.progress_text = Text("", size=12)
        
        # === OUTPUT ===
        self.result_image = Image(
            src="",
            width=512,
            height=512,
            fit=ft.ImageFit.CONTAIN,
            visible=False,
        )
        self.result_info = Text("", size=11, color=Colors.GREY_400)
        
        # === NO MODEL BANNER ===
        self.no_model_banner = Container(
            content=Row([
                ft.Icon(Icons.WARNING_AMBER_ROUNDED, size=24, color=Colors.AMBER_400),
                Text("🎨 No image model loaded! Load a model from the header dropdown to generate images.", 
                     size=14, color=Colors.AMBER_300),
            ], spacing=10, alignment=MainAxisAlignment.CENTER),
            bgcolor=Colors.with_opacity(0.3, Colors.AMBER_900),
            padding=padding.all(12),
            border_radius=border_radius.all(8),
            visible=True,  # Show by default until model loads
        )
        
        # Main container that will hold the dynamic layout
        self.main_container = Container(expand=True)
        
        # Initialize slider labels
        self._update_slider_label(self.lora_strength_value_text, "LoRA {:.2f}", self.lora_strength.value)
        self._update_slider_label(self.steps_value_text, "{} steps", int(self.steps_slider.value))
        self._update_slider_label(self.guidance_value_text, "CFG {:.1f}", self.guidance_slider.value)
    
    def _on_prompt_change(self, e):
        """Handle prompt field change - update save button enabled state."""
        self._update_save_prompt_btn()
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _on_prompt_change(self, e):
        """Handle prompt field change - update save button enabled state."""
        self._update_save_prompt_btn()
    
    def _update_save_prompt_btn(self):
        """Update save prompt button enabled state based on prompt text."""
        has_prompt = self.prompt_field.value and self.prompt_field.value.strip()
        self.save_prompt_btn.disabled = not has_prompt
        try:
            self.page.update()
        except:
            pass
    
    def _generate(self, e):
        """Generate image."""
        prompt = self.prompt_field.value
        debug_print(f" Generate clicked! Prompt: '{prompt[:30] if prompt else 'EMPTY'}...'")
        
        if not prompt or not prompt.strip():
            self.page.snack_bar = SnackBar(content=Text("Please enter a prompt"))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Check if model is loaded
        current_model, current_key = get_current_model()
        debug_print(f" Model state: exists={current_model is not None}, key={current_key}, app_state={app_state.image_model_loaded}")
        
        if current_model is None:
            print("[DEBUG] No model loaded - showing error")
            self.page.snack_bar = SnackBar(
                content=Text("❌ No model loaded! Load a model first."),
                bgcolor=Colors.RED_900,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        print("[DEBUG] Starting generation...")
        app_state.is_generating = True
        app_state.cancel_requested = False
        
        # Use disabled instead of visible to avoid layout shifts
        self.generate_btn.disabled = True
        self.generate_btn.text = "⏳ Generating..."
        self.stop_btn.disabled = False
        self.progress_bar.value = 0
        self.progress_text.value = "Starting..."
        self.page.update()
        
        def do_generate():
            steps = int(self.steps_slider.value)
            lora_name = self.lora_dropdown.value
            lora_str = self.lora_strength.value
            
            def progress_cb(step, total):
                if app_state.cancel_requested:
                    return
                self.progress_bar.value = step / total
                self.progress_text.value = f"Step {step}/{total}"
                self.page.update()
            
            try:
                logger.info(f"Starting generation: '{prompt[:50]}...'")
                result = generate_image(
                    prompt=prompt,
                    negative_prompt=self.negative_field.value or "",
                    steps=steps,
                    guidance_scale=self.guidance_slider.value,
                    width=int(self.width_dropdown.value),
                    height=int(self.height_dropdown.value),
                    seed=int(self.seed_field.value or -1),
                    lora_name=lora_name if lora_name != "None" else None,
                    lora_strength=lora_str,
                    progress_callback=progress_cb,
                )
                
                app_state.is_generating = False
                self.generate_btn.disabled = False
                self.generate_btn.text = "🎨 Generate"
                self.stop_btn.disabled = True
                self.progress_bar.value = 0
                self.progress_text.value = ""
                
                if result and not app_state.cancel_requested:
                    self.current_result = result
                    self.result_image.src_base64 = result.base64_data
                    self.result_image.visible = True
                    self.result_info.value = f"Seed: {result.seed} | Time: {result.generation_time:.1f}s"
                    if result.filepath:
                        self.result_info.value += f" | Saved: {Path(result.filepath).name}"
                    logger.info(f"Generation complete: seed={result.seed}")
                    # Rebuild layout to size image properly
                    self._rebuild_layout()
                elif not result:
                    logger.error("Generation returned None")
                    self.page.snack_bar = SnackBar(
                        content=Text("❌ Generation failed - check terminal for details"),
                        bgcolor=Colors.RED_900,
                    )
                    self.page.snack_bar.open = True
                
                self.page.update()
                
            except Exception as ex:
                logger.error(f"Generation exception: {ex}", exc_info=True)
                app_state.is_generating = False
                self.generate_btn.disabled = False
                self.generate_btn.text = "🎨 Generate"
                self.stop_btn.disabled = True
                self.progress_bar.value = 0
                self.progress_text.value = ""
                
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Error: {str(ex)[:100]}"),
                    bgcolor=Colors.RED_900,
                )
                self.page.snack_bar.open = True
                self.page.update()
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _stop_generation(self, e):
        """Cancel generation."""
        app_state.cancel_requested = True
        self.progress_text.value = "Cancelling..."
        self.page.update()
    
    def _save_to_library(self, e):
        """Save current prompt and settings to prompt library."""
        prompt = self.prompt_field.value.strip()
        if not prompt:
            self.page.snack_bar = SnackBar(
                content=Text("Enter a prompt first!"),
                bgcolor=Colors.ORANGE_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        if self.on_save_prompt:
            self.on_save_prompt(
                prompt=prompt,
                negative=self.negative_field.value.strip(),
                settings={
                    "steps": int(self.steps_slider.value),
                    "guidance": self.guidance_slider.value,
                    "width": int(self.width_dropdown.value),
                    "height": int(self.height_dropdown.value),
                    "lora": self.lora_dropdown.value,
                    "lora_strength": self.lora_strength.value,
                }
            )
        else:
            self.page.snack_bar = SnackBar(
                content=Text("Prompt library not connected."),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _build_settings_panel(self) -> Container:
        """Build the compact settings panel."""
        return Container(
            content=Column([
                Row([self.layout_dropdown], alignment=MainAxisAlignment.START),
                Container(height=10),
                Text("🎭 LoRA", weight=FontWeight.BOLD, size=13),
                self.lora_dropdown,
                # LoRA Strength slider
                Text("Strength:", size=11),
                self.lora_strength,
                self.lora_strength_value_text,
                Container(height=8),
                Text("⚙️ Settings", weight=FontWeight.BOLD, size=13),
                # Steps slider
                Text("Steps:", size=11),
                self.steps_slider,
                self.steps_value_text,
                Container(height=8),
                # CFG slider
                Text("CFG:", size=11),
                self.guidance_slider,
                self.guidance_value_text,
                Container(height=8),
                # Width & Height dropdowns
                self.width_dropdown,
                self.height_dropdown,
                Container(height=8),
                self.seed_field,
            ], spacing=3, scroll=ScrollMode.AUTO),
            width=200,
            padding=padding.all(12),
            bgcolor=Colors.with_opacity(0.15, Colors.ON_SURFACE),
            border_radius=border_radius.all(8),
        )
    
    def _build_prompt_section(self) -> Column:
        """Build the prompts and generate buttons section."""
        return Column([
            Row([
                Container(content=self.prompt_field, expand=True),
                self.save_prompt_btn,
            ], spacing=5, vertical_alignment=CrossAxisAlignment.START),
            self.negative_field,
            # Spacer pushes buttons to bottom
            Container(expand=True),
            # Buttons at bottom
            Row([
                self.generate_btn,
                self.stop_btn,
                Container(width=20),
                self.progress_bar,
                self.progress_text,
            ], alignment=MainAxisAlignment.START, vertical_alignment=CrossAxisAlignment.CENTER),
        ], spacing=8, expand=True)
    
    def _build_image_output(self, max_width: int = None, max_height: int = None) -> Container:
        """Build the image output container."""
        # Update image size based on current dimensions
        img_width = int(self.width_dropdown.value)
        img_height = int(self.height_dropdown.value)
        
        # Scale if needed to fit container
        if max_width and img_width > max_width:
            scale = max_width / img_width
            img_width = max_width
            img_height = int(img_height * scale)
        if max_height and img_height > max_height:
            scale = max_height / img_height
            img_height = max_height
            img_width = int(img_width * scale)
        
        self.result_image.width = img_width
        self.result_image.height = img_height
        
        return Container(
            content=Column([
                self.result_image,
                self.result_info,
            ], horizontal_alignment=CrossAxisAlignment.CENTER, spacing=5),
            bgcolor=Colors.with_opacity(0.15, Colors.ON_SURFACE),
            padding=padding.all(10),
            border_radius=border_radius.all(8),
            alignment=ft.alignment.center,
        )
    
    def _build_side_layout(self) -> Row:
        """Build layout with image on the right side - image gets more space."""
        return Row([
            # Left: Settings (compact)
            self._build_settings_panel(),
            # Middle: Prompts (fixed narrow width, stretches to match sidebar)
            Container(
                content=self._build_prompt_section(),
                width=350,
                padding=padding.only(left=15, right=15),
            ),
            # Right: Image output (expands to fill remaining space)
            Container(
                content=self._build_image_output(max_height=600),
                expand=True,
                alignment=ft.alignment.center,
            ),
        ], expand=True, spacing=0, vertical_alignment=CrossAxisAlignment.STRETCH)
    
    def _build_bottom_layout(self) -> Column:
        """Build layout with image on the bottom."""
        # Get sidebar with fixed height for bottom layout
        sidebar = Container(
            content=Column([
                Row([self.layout_dropdown], alignment=MainAxisAlignment.START),
                Container(height=10),
                Text("🎭 LoRA", weight=FontWeight.BOLD, size=13),
                self.lora_dropdown,
                Row([Text("Strength:", size=11, width=60), self.lora_strength, self.lora_strength_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                Container(height=10),
                Text("⚙️ Settings", weight=FontWeight.BOLD, size=13),
                Row([Text("Steps:", size=11, width=50), self.steps_slider, self.steps_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                Row([Text("CFG:", size=11, width=50), self.guidance_slider, self.guidance_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                Row([Text("Width:", size=11, width=50), self.width_slider, self.width_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                Row([Text("Height:", size=11, width=50), self.height_slider, self.height_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                self.seed_field,
            ], spacing=5),
            width=200,
            height=450,  # Fixed height for bottom layout
            padding=padding.all(12),
            bgcolor=Colors.with_opacity(0.15, Colors.ON_SURFACE),
            border_radius=border_radius.all(8),
        )
        
        # Prompt section with matching fixed height
        prompt_section = Container(
            content=Column([
                Container(content=self.prompt_field, expand=3),
                Container(content=self.negative_field, expand=2),
                Row([
                    self.generate_btn,
                    self.stop_btn,
                    Container(width=20),
                    self.progress_bar,
                    self.progress_text,
                ], alignment=MainAxisAlignment.START, vertical_alignment=CrossAxisAlignment.CENTER),
            ], spacing=8),
            height=450,  # Match sidebar height
            expand=True,
            padding=padding.only(left=15),
        )
        
        return Column([
            # Top row: Settings + Prompts
            Row([
                sidebar,
                prompt_section,
            ]),
            # Bottom: Image output (centered, full width)
            Container(
                content=self._build_image_output(max_width=800),
                expand=True,
                alignment=ft.alignment.center,
                padding=padding.only(top=15),
            ),
        ], expand=True, spacing=0)
    
    def build(self) -> Container:
        """Build the tab layout with dynamic image placement."""
        # Initialize with the appropriate layout
        layout = self._get_optimal_layout()
        if layout == self.LAYOUT_RIGHT:
            self.main_container.content = self._build_side_layout()
        else:
            self.main_container.content = self._build_bottom_layout()
        
        # Update initial state
        has_model = app_state.loaded_model_type == "image"
        self.generate_btn.disabled = not has_model
        self.no_model_banner.visible = not has_model
        
        return Container(
            content=Column([
                self.no_model_banner,
                self.main_container,
            ], spacing=10, expand=True),
            padding=padding.all(15),
            expand=True,
        )


# =============================================================================
# GALLERY TAB
# =============================================================================
class GalleryTab:
    """Gallery Tab - view generated images with list (paginated) and grid views."""
    
    GRID_COLUMNS = 4
    VIEW_LIST = "list"
    VIEW_GRID = "grid"
    PAGE_SIZE_OPTIONS = [10, 20, 50, 100]
    
    def __init__(self, page: Page):
        self.page = page
        self.current_page = 0
        self.page_size = 20  # Default page size
        self.total_pages = 1
        self.all_images: List[Dict] = []
        self.selected_images: Set[str] = set()
        self.current_view = self.VIEW_LIST  # Default to list view
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        # View toggle buttons
        self.list_view_btn = IconButton(
            icon=Icons.VIEW_LIST,
            tooltip="List View (paginated)",
            on_click=lambda e: self._switch_view(self.VIEW_LIST),
            selected=True,
            style=ButtonStyle(bgcolor={ControlState.SELECTED: Colors.PRIMARY}),
        )
        self.grid_view_btn = IconButton(
            icon=Icons.GRID_VIEW,
            tooltip="Grid View (scroll)",
            on_click=lambda e: self._switch_view(self.VIEW_GRID),
        )
        
        # Page size selector (only for list view)
        self.page_size_dropdown = Dropdown(
            label="Per Page",
            width=100,
            options=[dropdown.Option(str(size)) for size in self.PAGE_SIZE_OPTIONS],
            value="20",
            on_change=self._on_page_size_change,
        )
        
        # Total images count
        self.total_count_text = Text("Total: 0 images", size=12, color=Colors.GREY_400)
        
        # List view container
        self.image_list = ListView(spacing=10, padding=padding.all(10), expand=True)
        
        # Grid view container
        self.image_grid = GridView(
            runs_count=self.GRID_COLUMNS,
            max_extent=200,
            child_aspect_ratio=0.85,
            spacing=10,
            run_spacing=10,
            expand=True,
            padding=padding.all(10),
        )
        
        # Container that holds the current view
        self.view_container = Container(content=self.image_list, expand=True)
        
        # Pagination controls (only for list view)
        self.page_label = Text("Page 1 of 1", size=12)
        self.prev_btn = IconButton(
            icon=Icons.CHEVRON_LEFT,
            tooltip="Previous page",
            on_click=self._prev_page,
            disabled=True,
            style=ButtonStyle(
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
            ),
        )
        self.next_btn = IconButton(
            icon=Icons.CHEVRON_RIGHT,
            tooltip="Next page",
            on_click=self._next_page,
            disabled=True,
            style=ButtonStyle(
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
            ),
        )
        self.pagination_row = Row([
            self.prev_btn,
            self.page_label,
            self.page_size_dropdown,
            self.next_btn,
        ], alignment=MainAxisAlignment.CENTER)
        
        # Action buttons
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh gallery",
            on_click=self._refresh,
        )
        self.select_all_btn = TextButton(
            "Select All",
            on_click=self._toggle_select_all,
        )
        self.delete_selected_btn = ElevatedButton(
            "🗑️ Delete Selected",
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._delete_selected,
            visible=False,
        )
        self.download_selected_btn = ElevatedButton(
            "⬇️ Download Selected",
            bgcolor=Colors.GREEN_700,
            color=Colors.WHITE,
            on_click=self._download_selected,
            visible=False,
        )
        self.selection_count = Text("", size=12, color=Colors.GREY_400)
    
    def _switch_view(self, view_type: str):
        """Switch between list and grid views."""
        self.current_view = view_type
        self.list_view_btn.selected = (view_type == self.VIEW_LIST)
        self.grid_view_btn.selected = (view_type == self.VIEW_GRID)
        
        if view_type == self.VIEW_LIST:
            self.view_container.content = self.image_list
            self.pagination_row.visible = True
            self._update_list()
        else:
            self.view_container.content = self.image_grid
            self.pagination_row.visible = False
            self._update_grid()
        
        self.page.update()
    
    def _refresh(self, e=None):
        """Refresh the gallery."""
        self.current_page = 0
        self.selected_images.clear()
        self._load_all_images()
        if self.current_view == self.VIEW_LIST:
            self._update_list()
        else:
            self._update_grid()
        self.page.update()
    
    def _on_page_size_change(self, e):
        """Handle page size dropdown change."""
        self.page_size = int(self.page_size_dropdown.value)
        self.current_page = 0  # Reset to first page
        self._recalculate_pages()
        self._update_list()
        self.page.update()
    
    def _load_all_images(self):
        """Load all images from disk."""
        self.all_images = get_generated_images(limit=500)
        self._recalculate_pages()
        self.total_count_text.value = f"Total: {len(self.all_images)} images"
    
    def _recalculate_pages(self):
        """Recalculate total pages based on current page size."""
        self.total_pages = max(1, (len(self.all_images) + self.page_size - 1) // self.page_size)
    
    def _get_current_page_images(self) -> List[Dict]:
        """Get images for current page (list view)."""
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.all_images[start:end]
    
    def _prev_page(self, e=None):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_list()
            self.page.update()
    
    def _next_page(self, e=None):
        """Go to next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_list()
            self.page.update()
    
    def _toggle_select_all(self, e=None):
        """Toggle select all images on current view."""
        if self.current_view == self.VIEW_LIST:
            page_images = self._get_current_page_images()
        else:
            page_images = self.all_images
        
        page_paths = {img["path"] for img in page_images}
        
        if page_paths.issubset(self.selected_images):
            self.selected_images -= page_paths
        else:
            self.selected_images.update(page_paths)
        
        if self.current_view == self.VIEW_LIST:
            self._update_list()
        else:
            self._update_grid()
        self.page.update()
    
    def _show_fullscreen_image(self, filepath: str, img_info: dict):
        """Show image in fullscreen popup."""
        # Create close button
        close_btn = IconButton(
            icon=Icons.CLOSE,
            icon_size=24,
        )
        
        # Create image viewer
        fullscreen_image = Image(
            src=filepath,
            fit=ft.ImageFit.CONTAIN,
            expand=True,
        )
        
        # Build info sections
        info_controls = [
            Text(f"📁 {img_info.get('filename', 'Image')}", weight=FontWeight.BOLD, size=14),
            Text(f"🎨 Model: {img_info.get('model', 'Unknown')}", size=12),
            Text(f"🌱 Seed: {img_info.get('seed', 'N/A')}", size=12),
            Text(f"⏰ {img_info.get('timestamp', '')}", size=11, color=Colors.GREY_400),
        ]
        
        if img_info.get('prompt'):
            info_controls.extend([
                Text("✨ Prompt:", size=12, weight=FontWeight.BOLD),
                Text(img_info.get('prompt', ''), size=11, color=Colors.GREY_300, selectable=True),
            ])
        
        if img_info.get('negative'):
            info_controls.extend([
                Text("🚫 Negative:", size=12, weight=FontWeight.BOLD),
                Text(img_info.get('negative', ''), size=11, color=Colors.GREY_300, selectable=True),
            ])
        
        info_text = Column(info_controls, spacing=10, scroll=ScrollMode.AUTO)
        
        # Create buttons row
        buttons = Row([
            IconButton(
                icon=Icons.CONTENT_COPY,
                tooltip="Copy Prompt",
                on_click=lambda e: self._copy_prompt(img_info.get('prompt', '')),
            ),
            IconButton(
                icon=Icons.FOLDER_OPEN,
                tooltip="Open Folder",
                on_click=lambda e: self._open_folder(filepath),
            ),
            IconButton(
                icon=Icons.DELETE_OUTLINE,
                tooltip="Delete",
                icon_color=Colors.RED_400,
                on_click=lambda e: self._handle_fullscreen_delete(filepath),
            ),
        ], spacing=10)
        
        # Create main content
        content = Row([
            Container(
                content=fullscreen_image,
                expand=2,
                bgcolor=Colors.with_opacity(0.3, Colors.BLACK),
                border_radius=border_radius.all(8),
            ),
            Container(
                content=Column([
                    Row([
                        Text("Image Details", weight=FontWeight.BOLD, size=14),
                        Container(expand=True),
                        close_btn,
                    ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                    Divider(),
                    info_text,
                    buttons,
                ], spacing=10),
                expand=1,
                padding=padding.all(15),
            ),
        ], spacing=15, expand=True)
        
        dialog = AlertDialog(
            content=content,
            modal=True,
            inset_padding=50,
        )
        
        close_btn.on_click = lambda e: self._close_fullscreen(dialog)
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _close_fullscreen(self, dialog):
        """Close the fullscreen image viewer."""
        try:
            dialog.open = False
            if dialog in self.page.overlay:
                self.page.overlay.remove(dialog)
            self.page.update()
        except Exception as e:
            logger.error(f"Error closing fullscreen dialog: {e}")
            try:
                self.page.update()
            except:
                pass
    
    def _handle_fullscreen_delete(self, filepath: str):
        """Delete image from fullscreen view."""
        self._delete_image(filepath)
        # Close all dialogs
        try:
            for dialog in list(self.page.overlay):
                if isinstance(dialog, AlertDialog):
                    try:
                        dialog.open = False
                        if dialog in self.page.overlay:
                            self.page.overlay.remove(dialog)
                    except:
                        pass
            self.page.update()
        except Exception as e:
            logger.error(f"Error closing dialogs: {e}")
    
    def _toggle_selection(self, filepath: str):
        """Toggle selection of a single image."""
        if filepath in self.selected_images:
            self.selected_images.discard(filepath)
        else:
            self.selected_images.add(filepath)
        self._update_selection_ui()
        # Update the current view to reflect checkbox state change
        if self.current_view == self.VIEW_LIST:
            self._update_list()
        else:
            self._update_grid()
        self.page.update()
    
    def _update_selection_ui(self):
        """Update selection count and button visibility."""
        count = len(self.selected_images)
        self.selection_count.value = f"{count} selected" if count > 0 else ""
        self.delete_selected_btn.visible = count > 0
        self.download_selected_btn.visible = count > 0
    
    def _delete_selected(self, e=None):
        """Delete all selected images with confirmation."""
        count = len(self.selected_images)
        if count == 0:
            return
        
        def do_delete(e):
            deleted = 0
            for filepath in list(self.selected_images):
                try:
                    Path(filepath).unlink()
                    deleted += 1
                except:
                    pass
            self.selected_images.clear()
            self._refresh()
            self.page.snack_bar = SnackBar(content=Text(f"🗑️ Deleted {deleted} images"))
            self.page.snack_bar.open = True
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Selected Images?"),
            content=Text(f"Are you sure you want to delete {count} selected images?\nThis cannot be undone."),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Delete All", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _download_selected(self, e=None):
        """Download all selected images to a Downloads folder."""
        count = len(self.selected_images)
        if count == 0:
            return
        
        def do_download(e):
            import shutil
            # Create Downloads folder in user's home directory
            downloads_dir = Path.home() / "Downloads" / "KVGenius_Images"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded = 0
            for filepath in list(self.selected_images):
                try:
                    src = Path(filepath)
                    if src.exists():
                        dst = downloads_dir / src.name
                        shutil.copy2(src, dst)
                        downloaded += 1
                except Exception as e:
                    print(f"Failed to download {filepath}: {e}")
            
            self.page.snack_bar = SnackBar(
                content=Text(f"✅ Downloaded {downloaded} images to {downloads_dir}")
            )
            self.page.snack_bar.open = True
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Download Selected Images?"),
            content=Text(f"Download {count} selected images to Downloads/KVGenius_Images?"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Download", on_click=do_download),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _open_folder(self, filepath: str):
        """Open the folder containing the image."""
        import subprocess
        folder = Path(filepath).parent
        if sys.platform == "win32":
            subprocess.run(["explorer", str(folder)])
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])
    
    def _copy_prompt(self, prompt: str):
        """Copy prompt to clipboard."""
        self.page.set_clipboard(prompt)
        self.page.snack_bar = SnackBar(content=Text("📋 Prompt copied to clipboard!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _delete_image(self, filepath: str):
        """Delete an image with confirmation."""
        def do_delete(e):
            try:
                Path(filepath).unlink()
                self.selected_images.discard(filepath)
                self._refresh()
                self.page.snack_bar = SnackBar(content=Text("🗑️ Image deleted"))
                self.page.snack_bar.open = True
            except Exception as ex:
                self.page.snack_bar = SnackBar(content=Text(f"❌ Error: {ex}"))
                self.page.snack_bar.open = True
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Image?"),
            content=Text(f"Are you sure you want to delete this image?\n{Path(filepath).name}"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Delete", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _show_metadata(self, img_info: Dict):
        """Show image metadata in a dialog."""
        content = Column([
            Text(f"📄 File: {img_info['filename']}", selectable=True),
            Text(f"🎨 Model: {img_info.get('model', 'Unknown')}", selectable=True),
            Text(f"🎲 Seed: {img_info.get('seed', 'Unknown')}", selectable=True),
            Text(f"📝 Prompt:", weight=FontWeight.BOLD),
            Container(
                content=Text(img_info.get('prompt', 'No prompt'), selectable=True, size=12),
                bgcolor=Colors.with_opacity(0.1, Colors.ON_SURFACE),
                padding=padding.all(8),
                border_radius=border_radius.all(5),
            ),
            Text(f"⏰ Created: {img_info.get('timestamp', '')}", size=11),
        ], spacing=8, scroll=ScrollMode.AUTO)
        
        def close(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Image Details"),
            content=Container(content=content, width=400, height=300),
            actions=[TextButton("Close", on_click=close)],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _update_list(self):
        """Update the list view with current page images."""
        self.image_list.controls.clear()
        page_images = self._get_current_page_images()
        
        if not page_images:
            self.image_list.controls.append(
                Text("No generated images yet", color=Colors.GREY_500)
            )
        else:
            for img_info in page_images:
                filepath = img_info["path"]
                prompt = img_info.get("prompt", "")
                is_selected = filepath in self.selected_images
                
                # Action buttons
                actions = Row([
                    Checkbox(
                        value=is_selected,
                        on_change=lambda e, p=filepath: self._toggle_selection(p),
                    ),
                    IconButton(
                        icon=Icons.FULLSCREEN,
                        tooltip="View Fullscreen",
                        icon_size=18,
                        on_click=lambda e, p=filepath, info=img_info: self._show_fullscreen_image(p, info),
                    ),
                    IconButton(
                        icon=Icons.FOLDER_OPEN,
                        tooltip="Open Folder",
                        icon_size=18,
                        on_click=lambda e, p=filepath: self._open_folder(p),
                    ),
                    IconButton(
                        icon=Icons.CONTENT_COPY,
                        tooltip="Copy Prompt",
                        icon_size=18,
                        on_click=lambda e, pr=prompt: self._copy_prompt(pr),
                    ),
                    IconButton(
                        icon=Icons.INFO_OUTLINE,
                        tooltip="View Details",
                        icon_size=18,
                        on_click=lambda e, info=img_info: self._show_metadata(info),
                    ),
                    IconButton(
                        icon=Icons.DELETE_OUTLINE,
                        tooltip="Delete",
                        icon_size=18,
                        icon_color=Colors.RED_400,
                        on_click=lambda e, p=filepath: self._delete_image(p),
                    ),
                ], spacing=0)
                
                card = Card(
                    content=Container(
                        content=Row([
                            Image(
                                src=filepath,
                                width=120,
                                height=120,
                                fit=ft.ImageFit.COVER,
                                border_radius=border_radius.all(8),
                            ),
                            Column([
                                Text(img_info["filename"], weight=FontWeight.BOLD, size=13),
                                Text(f"Model: {img_info['model']}", size=11, color=Colors.GREY_400),
                                Text(f"Seed: {img_info['seed']}", size=11, color=Colors.GREY_400),
                                Text(img_info["timestamp"], size=10, color=Colors.GREY_600),
                                actions,
                            ], expand=True, spacing=3),
                        ], spacing=15),
                        padding=padding.all(10),
                        bgcolor=Colors.with_opacity(0.2, Colors.PRIMARY) if is_selected else None,
                    ),
                )
                self.image_list.controls.append(card)
        
        # Update pagination controls
        self.page_label.value = f"Page {self.current_page + 1} of {self.total_pages}"
        self.prev_btn.disabled = self.current_page <= 0
        self.next_btn.disabled = self.current_page >= self.total_pages - 1
        self._update_selection_ui()
    
    def _update_grid(self):
        """Update the grid view with ALL images (no pagination)."""
        self.image_grid.controls.clear()
        
        if not self.all_images:
            self.image_grid.controls.append(
                Container(
                    content=Text("No generated images yet", color=Colors.GREY_500),
                    alignment=alignment.center,
                )
            )
        else:
            for img_info in self.all_images:
                filepath = img_info["path"]
                is_selected = filepath in self.selected_images
                
                # Create image card with checkbox overlay
                card = Container(
                    content=Stack([
                        # Image
                        Container(
                            content=Image(
                                src=filepath,
                                fit=ft.ImageFit.COVER,
                                border_radius=border_radius.all(8),
                            ),
                            border_radius=border_radius.all(8),
                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                        ),
                        # Selection overlay
                        Container(
                            bgcolor=Colors.with_opacity(0.3, Colors.PRIMARY) if is_selected else None,
                            border=border.all(3, Colors.PRIMARY) if is_selected else None,
                            border_radius=border_radius.all(8),
                        ),
                        # Checkbox in top-left
                        Container(
                            content=Checkbox(
                                value=is_selected,
                                on_change=lambda e, p=filepath: self._toggle_selection(p),
                            ),
                            alignment=alignment.top_left,
                            padding=padding.only(left=5, top=5),
                        ),
                        # Info overlay at bottom
                        Container(
                            content=Container(
                                content=Row([
                                    IconButton(
                                        icon=Icons.FULLSCREEN,
                                        tooltip="Fullscreen",
                                        icon_size=16,
                                        icon_color=Colors.WHITE,
                                        on_click=lambda e, p=filepath, info=img_info: self._show_fullscreen_image(p, info),
                                    ),
                                    IconButton(
                                        icon=Icons.INFO_OUTLINE,
                                        tooltip="Details",
                                        icon_size=16,
                                        icon_color=Colors.WHITE,
                                        on_click=lambda e, info=img_info: self._show_metadata(info),
                                    ),
                                    IconButton(
                                        icon=Icons.CONTENT_COPY,
                                        tooltip="Copy Prompt",
                                        icon_size=16,
                                        icon_color=Colors.WHITE,
                                        on_click=lambda e, pr=img_info.get("prompt", ""): self._copy_prompt(pr),
                                    ),
                                    IconButton(
                                        icon=Icons.DELETE_OUTLINE,
                                        tooltip="Delete",
                                        icon_size=16,
                                        icon_color=Colors.RED_300,
                                        on_click=lambda e, p=filepath: self._delete_image(p),
                                    ),
                                ], spacing=0, alignment=MainAxisAlignment.CENTER),
                                bgcolor=Colors.with_opacity(0.7, Colors.BLACK),
                                border_radius=border_radius.only(bottom_left=8, bottom_right=8),
                                padding=padding.symmetric(vertical=2),
                            ),
                            alignment=alignment.bottom_center,
                        ),
                    ]),
                    border_radius=border_radius.all(8),
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    on_click=lambda e, p=filepath: self._toggle_selection(p),
                )
                self.image_grid.controls.append(card)
        
        self._update_selection_ui()
    
    def build(self) -> Container:
        """Build the gallery tab."""
        self._load_all_images()
        self._update_list()  # Start with list view
        
        # Header with view toggle and actions
        header = Row([
            Row([
                Text("🖼️ Gallery", size=18, weight=FontWeight.BOLD),
                Container(width=20),
                self.list_view_btn,
                self.grid_view_btn,
                self.total_count_text,
            ]),
            Row([
                self.selection_count,
                self.download_selected_btn,
                self.delete_selected_btn,
                self.select_all_btn,
                self.refresh_btn,
            ], spacing=10),
        ], alignment=MainAxisAlignment.SPACE_BETWEEN)
        
        return Container(
            content=Column([
                header,
                self.view_container,
                self.pagination_row,
            ], spacing=10),
            padding=padding.all(15),
            expand=True,
        )


# =============================================================================
# MODEL MANAGER TAB
# =============================================================================
class ModelManagerTab:
    """Model Manager - view and download all models (image + chat)."""
    
    def __init__(self, page: Page):
        self.page = page
        self._downloading: Dict[str, bool] = {}  # Track downloads in progress
        self._download_progress: Dict[str, float] = {}  # repo_id -> progress (0.0-1.0)
        self._download_status: Dict[str, str] = {}  # repo_id -> status message
        self._download_queue: List[Dict] = []  # Queue of pending/active downloads
        # Store references to progress UI controls for lightweight updates
        self._progress_controls: Dict[str, Dict] = {}  # repo_id -> {bar, text, pct_text, dl_bar, dl_text, dl_status}
        self.model_cache = Path(__file__).parent / "data" / "model_cache"
        self._build_ui()
    
    def _build_ui(self):
        self.image_model_list = ListView(spacing=10, padding=20, expand=True)
        self.chat_model_list = ListView(spacing=10, padding=20, expand=True)
        self.downloads_list = ListView(spacing=10, padding=20, expand=True)
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh model list",
            on_click=lambda e: self._refresh(),
        )
        self.status_text = Text("", size=11, color=Colors.GREY_400)
    
    def _refresh(self):
        """Refresh the model list. Also clears completed/failed downloads from the queue."""
        self._download_queue = [item for item in self._download_queue if item["status"] == "downloading"]
        self._load_models()
        self._refresh_downloads_tab()
        if self.page:
            self.page.update()
    
    def _download_model(self, model_name: str, repo_id: str, model_type: str):
        """Start downloading a model in background."""
        if repo_id in self._downloading and self._downloading[repo_id]:
            return  # Already downloading
        
        self._downloading[repo_id] = True
        self._download_progress[repo_id] = 0.0
        self._download_status[repo_id] = "Starting..."
        
        # Add to download queue
        queue_item = {
            "name": model_name,
            "repo_id": repo_id,
            "model_type": model_type,
            "status": "downloading",
        }
        self._download_queue.append(queue_item)
        self._load_models()
        self._refresh_downloads_tab()
        self.status_text.value = f"Downloading {model_name}..."
        self.page.update()
        
        def do_download():
            def progress_callback(pct: float, msg: str):
                self._download_progress[repo_id] = pct
                self._download_status[repo_id] = msg
                self.status_text.value = f"[{pct*100:.0f}%] {msg}"
                # Lightweight update: just update existing controls directly
                self._update_progress_ui(repo_id, pct, msg)
                try:
                    self.page.update()
                except:
                    pass
            
            # Use appropriate download function
            if model_type == "image":
                success, message = download_model(repo_id, progress_callback)
            else:
                success, message = download_chat_model(repo_id, progress_callback)
            
            self._downloading[repo_id] = False
            self._download_progress.pop(repo_id, None)
            self._download_status.pop(repo_id, None)
            self._progress_controls.pop(repo_id, None)
            
            # Update queue item status
            for item in self._download_queue:
                if item["repo_id"] == repo_id:
                    item["status"] = "completed" if success else "failed"
                    break
            
            if success:
                self.status_text.value = f"✅ {model_name} downloaded!"
            else:
                self.status_text.value = f"❌ {message}"
            
            # Refresh everything
            self._load_models()
            self._refresh_downloads_tab()
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=do_download, daemon=True).start()
    
    def _open_folder(self, folder_path: str):
        """Open a folder in the system file explorer."""
        import subprocess
        try:
            p = Path(folder_path)
            # If it's a file, open its parent folder
            if p.is_file():
                subprocess.run(["explorer", "/select,", str(p)])
            elif p.is_dir():
                subprocess.run(["explorer", str(p)])
            else:
                # Path doesn't exist, open parent
                parent = p.parent
                if parent.exists():
                    subprocess.run(["explorer", str(parent)])
        except Exception as e:
            logger.error(f"Error opening folder: {e}")
    
    def _refresh_downloads_tab(self):
        """Refresh the downloads tab with current queue status."""
        self.downloads_list.controls.clear()
        
        if not self._download_queue:
            self.downloads_list.controls.append(
                Container(
                    content=Text(
                        "No downloads yet. Go to Image Models or Chat Models tab to download models.",
                        size=12, color=Colors.GREY_500, text_align=ft.TextAlign.CENTER,
                    ),
                    padding=padding.all(40),
                    alignment=alignment.center,
                )
            )
            return
        
        # Show downloads in reverse order (newest first)
        for item in reversed(self._download_queue):
            repo_id = item["repo_id"]
            name = item["name"]
            status = item["status"]
            progress = self._download_progress.get(repo_id, 0.0)
            status_msg = self._download_status.get(repo_id, "")
            
            if status == "downloading":
                icon = Icons.DOWNLOADING
                icon_color = Colors.BLUE_400
                status_label = f"Downloading... {progress*100:.0f}%"
                bar_color = Colors.BLUE_400
            elif status == "completed":
                icon = Icons.CHECK_CIRCLE
                icon_color = Colors.GREEN_400
                status_label = "Completed"
                bar_color = Colors.GREEN_400
                progress = 1.0
            else:
                icon = Icons.ERROR
                icon_color = Colors.RED_400
                status_label = "Failed"
                bar_color = Colors.RED_400
            
            dl_status_text = Text(status_msg if status == "downloading" else status_label,
                                  size=11, color=Colors.GREY_400)
            dl_label_text = Text(status_label, size=11, color=icon_color)
            dl_bar = ProgressBar(
                value=progress,
                color=bar_color,
                bgcolor=Colors.GREY_800,
                height=6,
                border_radius=border_radius.all(3),
            )
            
            card = Card(
                content=Container(
                    content=Column([
                        Row([
                            ft.Icon(icon, color=icon_color, size=24),
                            Column([
                                Text(name, weight=FontWeight.BOLD, size=13),
                                dl_status_text,
                            ], expand=True, spacing=2),
                            dl_label_text,
                        ], spacing=10),
                        dl_bar,
                    ], spacing=8),
                    padding=padding.all(12),
                ),
            )
            self.downloads_list.controls.append(card)
            
            # Store download tab control references for lightweight updates
            if status == "downloading":
                ctrl = self._progress_controls.get(repo_id, {})
                ctrl["dl_bar"] = dl_bar
                ctrl["dl_text"] = dl_label_text
                ctrl["dl_status"] = dl_status_text
                self._progress_controls[repo_id] = ctrl
    
    def _update_progress_ui(self, repo_id: str, pct: float, msg: str):
        """Lightweight progress update - directly update existing controls without rebuilding."""
        ctrls = self._progress_controls.get(repo_id)
        if not ctrls:
            return
        
        pct_str = f"{pct*100:.0f}%"
        
        # Update model card controls
        if "bar" in ctrls:
            ctrls["bar"].value = pct
        if "text" in ctrls:
            ctrls["text"].value = f"{pct_str} - {msg}"
        if "pct_text" in ctrls:
            ctrls["pct_text"].value = f"Downloading... {pct_str}"
        
        # Update downloads tab controls
        if "dl_bar" in ctrls:
            ctrls["dl_bar"].value = pct
        if "dl_text" in ctrls:
            ctrls["dl_text"].value = f"Downloading... {pct_str}"
        if "dl_status" in ctrls:
            ctrls["dl_status"].value = msg
    
    def _get_folder_size_gb(self, folder_path: Path) -> float:
        """Calculate total size of a folder in GB."""
        total_size = 0
        try:
            if folder_path.exists():
                for path in folder_path.rglob('*'):
                    if path.is_file():
                        total_size += path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating folder size: {e}")
        return total_size / (1024 ** 3)  # Convert to GB
    
    def _confirm_delete_files(self, model_name: str, repo_id: str):
        """Show confirmation dialog for deleting downloaded model files."""
        # Calculate size for either local file or HF model
        repo_path = Path(repo_id)
        
        if repo_path.is_file() or (repo_path.parent.name == "checkpoints" and repo_id.endswith((".safetensors", ".ckpt"))):
            # Local checkpoint - get file size directly
            size_gb = repo_path.stat().st_size / (1024**3) if repo_path.exists() else 0
        else:
            # HuggingFace model - calculate folder size
            safe_name = "models--" + repo_id.replace("/", "--")
            model_path = self.model_cache / safe_name
            size_gb = self._get_folder_size_gb(model_path)
        
        def delete_files(e):
            self._delete_model_files(model_name, repo_id)
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text(f"Delete {model_name}?"),
            content=Container(
                content=Column([
                    Text(
                        f"This will delete the downloaded files and free up {size_gb:.2f} GB of storage.",
                        size=12,
                    ),
                    Text(
                        "⚠️  You can download it again anytime.",
                        size=11,
                        color=Colors.BLUE_400,
                    ),
                ], spacing=8),
                padding=padding.all(10),
            ),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton(
                    "Delete Files",
                    on_click=delete_files,
                    style=ButtonStyle(color=Colors.RED_400),
                ),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _delete_model_files(self, model_name: str, repo_id: str):
        """Delete the downloaded model files (allows re-download)."""
        try:
            # Check if it's a local checkpoint file (path) or HF model (repo_id)
            repo_path = Path(repo_id)
            
            if repo_path.is_file() or (repo_path.parent.name == "checkpoints" and repo_id.endswith((".safetensors", ".ckpt"))):
                # Local checkpoint - delete the actual file
                size_gb = repo_path.stat().st_size / (1024**3) if repo_path.exists() else 0
                if repo_path.exists():
                    repo_path.unlink()
                    self.status_text.value = f"🗑️  Deleted {model_name} ({size_gb:.2f} GB freed)"
                    logger.info(f"Deleted local checkpoint: {model_name}")
            else:
                # HuggingFace model - delete from model cache
                safe_name = "models--" + repo_id.replace("/", "--")
                model_path = self.model_cache / safe_name
                
                if model_path.exists():
                    size_gb = self._get_folder_size_gb(model_path)
                    shutil.rmtree(model_path)
                    self.status_text.value = f"🗑️  Deleted {model_name} ({size_gb:.2f} GB freed)"
                    logger.info(f"Deleted model files: {model_name}")
            
            self._load_models()
            self.page.update()
        except Exception as e:
            logger.error(f"Error deleting model files: {e}")
            self.status_text.value = f"❌ Error deleting: {e}"
            self.page.update()
    
    def _confirm_delete_local_model(self, model_name: str, file_path: str):
        """Show confirmation dialog for deleting a local model file and removing from config."""
        try:
            repo_path = Path(file_path)
            size_gb = repo_path.stat().st_size / (1024**3) if repo_path.exists() else 0
        except:
            size_gb = 0
        
        def delete_local(e):
            self._delete_local_model(model_name, file_path)
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Local Model?"),
            content=Container(
                content=Column([
                    Text(
                        f"This will permanently delete '{model_name}' from your computer.",
                        size=12,
                    ),
                    Text(
                        f"Storage freed: {size_gb:.2f} GB",
                        size=11,
                        color=Colors.YELLOW_400,
                        weight=FontWeight.BOLD,
                    ),
                    Text(
                        "⚠️  This cannot be undone. It will be removed from the list.",
                        size=11,
                        color=Colors.RED_400,
                    ),
                ], spacing=8),
                padding=padding.all(10),
            ),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton(
                    "Delete Permanently",
                    on_click=delete_local,
                    style=ButtonStyle(color=Colors.RED_700),
                ),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _delete_local_model(self, model_name: str, file_path: str):
        """Delete local model file from disk and remove from config."""
        try:
            repo_path = Path(file_path)
            size_gb = 0
            
            # Delete the actual file
            if repo_path.exists():
                size_gb = repo_path.stat().st_size / (1024**3)
                repo_path.unlink()
                logger.info(f"Deleted local model file: {model_name}")
            
            # Remove from config file
            self._remove_model_from_config(model_name)
            
            self.status_text.value = f"🗑️  Deleted {model_name} ({size_gb:.2f} GB freed)"
            self._load_models()
            self.page.update()
        except Exception as e:
            logger.error(f"Error deleting local model: {e}")
            self.status_text.value = f"❌ Error deleting: {e}"
            self.page.update()
    
    def _remove_model_from_config(self, model_name: str):
        """Remove a model from the image_model_presets.yaml config file."""
        try:
            config_path = Path(__file__).parent / "config" / "image_model_presets.yaml"
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return
            
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Look for model in both 'huggingface' and 'local' sections
            for section in ['huggingface', 'local']:
                if section in config.get('models', {}):
                    if model_name in config['models'][section]:
                        del config['models'][section][model_name]
                        logger.info(f"Removed {model_name} from {section} section in config")
                        break
            
            # Write updated config back
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Updated config file: removed {model_name}")
        except Exception as e:
            logger.warning(f"Could not update config file: {e}")
            # Don't fail the deletion if config update fails
    
    def _confirm_remove_model(self, model_name: str):
        """Show confirmation dialog for permanently removing a model from the list."""
        def remove_model(e):
            self._remove_model_permanently(model_name)
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Remove Model Permanently?"),
            content=Container(
                content=Column([
                    Text(
                        f"Remove '{model_name}' from the available models list?",
                        size=12,
                    ),
                    Text(
                        "⚠️  This cannot be undone. The model will no longer appear in the list.",
                        size=11,
                        color=Colors.RED_400,
                        weight=FontWeight.BOLD,
                    ),
                ], spacing=8),
                padding=padding.all(10),
            ),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton(
                    "Remove Permanently",
                    on_click=remove_model,
                    style=ButtonStyle(color=Colors.RED_700),
                ),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _remove_model_permanently(self, model_name: str):
        """Remove a model permanently from the list (not yet implemented - would need model config updates)."""
        # TODO: Implement by removing from config files or marking as hidden
        logger.warning(f"Model removal not yet fully implemented: {model_name}")
        self.status_text.value = f"Permanent removal not yet implemented for {model_name}"
        self.page.update()
    
    def _load_models(self):
        self.image_model_list.controls.clear()
        self.chat_model_list.controls.clear()
        
        # Image Models
        image_models = get_available_image_models()
        for name, info in image_models.items():
            is_hf = info.get("source") == "huggingface"
            repo_id = info.get("id", "")
            downloaded = is_model_downloaded(repo_id) if is_hf else True
            fully_downloaded = _is_model_fully_downloaded(repo_id) if is_hf else True
            is_partial = downloaded and not fully_downloaded and is_hf
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "image")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(self.image_model_list, name, info, "image", is_loaded, downloaded, is_downloading, is_hf, repo_id, is_partial)
        
        # Chat Models
        chat_models = get_available_chat_models()
        for name, info in chat_models.items():
            repo_id = info.get("id", "")
            downloaded = is_chat_model_downloaded(repo_id)
            fully_downloaded = _is_chat_model_fully_downloaded(repo_id)
            is_partial = downloaded and not fully_downloaded
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "chat")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(self.chat_model_list, name, info, "chat", is_loaded, downloaded, is_downloading, True, repo_id, is_partial)
    
    def _add_model_card(self, target_list: ListView, name: str, info: Dict, model_type: str, is_loaded: bool, 
                        downloaded: bool, is_downloading: bool, is_hf: bool, repo_id: str, is_partial: bool = False):
        """Add a model card to the list."""
        if is_loaded:
            status_icon = Icons.CHECK_CIRCLE
            status_color = Colors.GREEN_400
            status_text = "Loaded"
        elif is_partial and not is_downloading:
            status_icon = Icons.WARNING_AMBER
            status_color = Colors.ORANGE_400
            status_text = "Partial Download"
        elif downloaded:
            status_icon = Icons.DOWNLOAD_DONE
            status_color = Colors.YELLOW_400
            status_text = "Downloaded"
        elif is_downloading:
            status_icon = Icons.DOWNLOADING
            status_color = Colors.BLUE_400
            status_text = "Downloading..."
        else:
            status_icon = Icons.CLOUD_DOWNLOAD
            status_color = Colors.GREY_500
            status_text = "Not Downloaded"
        
        # Action buttons based on state
        action_buttons = []
        is_local = info.get("source") == "local"
        
        if is_downloading:
            progress = self._download_progress.get(repo_id, 0.0)
            pct_text_ctrl = Text(f"Downloading... {progress*100:.0f}%", size=11)
            action_buttons.append(
                Column([
                    Row([
                        ft.ProgressRing(width=16, height=16, stroke_width=2),
                        pct_text_ctrl,
                    ], spacing=5),
                ], spacing=3)
            )
        elif is_local:
            # Local checkpoint - allow delete (deletes from disk AND config)
            action_buttons.append(
                ElevatedButton(
                    "Delete Model",
                    icon=Icons.DELETE_OUTLINE,
                    on_click=lambda e, n=name, r=repo_id: self._confirm_delete_local_model(n, r),
                    style=ButtonStyle(color=Colors.RED_700),
                )
            )
        elif is_hf:
            # HuggingFace models
            if is_partial and not is_downloading:
                # Partially downloaded - show Resume and Delete buttons
                action_buttons.append(
                    ElevatedButton(
                        "Resume Download",
                        icon=Icons.PLAY_ARROW,
                        on_click=lambda e, n=name, r=repo_id, t=model_type: self._download_model(n, r, t),
                        style=ButtonStyle(color=Colors.ORANGE_400),
                    )
                )
                action_buttons.append(
                    ElevatedButton(
                        "Delete Files",
                        icon=Icons.DELETE_OUTLINE,
                        on_click=lambda e, n=name, r=repo_id: self._confirm_delete_files(n, r),
                        style=ButtonStyle(color=Colors.RED_700),
                    )
                )
            elif not downloaded:
                # Not downloaded - show Download and Remove buttons
                action_buttons.append(
                    ElevatedButton(
                        "Download",
                        icon=Icons.DOWNLOAD,
                        on_click=lambda e, n=name, r=repo_id, t=model_type: self._download_model(n, r, t),
                    )
                )
                action_buttons.append(
                    ElevatedButton(
                        "Remove Model",
                        icon=Icons.DELETE_FOREVER,
                        on_click=lambda e, n=name: self._confirm_remove_model(n),
                        style=ButtonStyle(color=Colors.RED_700),
                    )
                )
            else:
                # Downloaded - show Delete Files and Remove Model
                action_buttons.append(
                    ElevatedButton(
                        "Delete Files",
                        icon=Icons.DELETE_OUTLINE,
                        on_click=lambda e, n=name, r=repo_id: self._confirm_delete_files(n, r),
                        style=ButtonStyle(color=Colors.ORANGE_400),
                    )
                )
                action_buttons.append(
                    ElevatedButton(
                        "Remove Model",
                        icon=Icons.DELETE_FOREVER,
                        on_click=lambda e, n=name: self._confirm_remove_model(n),
                        style=ButtonStyle(color=Colors.RED_700),
                    )
                )
        
        # Type badge color
        type_text = info.get("type", "sd").upper() if model_type == "image" else "LLM"
        type_color = Colors.BLUE_900 if model_type == "image" else Colors.ORANGE_900
        
        # Build info column contents
        info_column_items = [
            Text(name, weight=FontWeight.BOLD),
            Text(info.get("description", ""), size=11, color=Colors.GREY_400),
        ]
        
        # For local models, show file path and open folder button
        if is_local:
            file_path = repo_id  # For local models, repo_id is the file path
            path_display = str(file_path) if len(str(file_path)) < 60 else f"...{str(file_path)[-57:]}"
            info_column_items.append(
                Row([
                    ft.Icon(Icons.FOLDER_OPEN, size=14, color=Colors.GREY_500),
                    Text(path_display, size=10, color=Colors.GREY_500),
                    IconButton(
                        icon=Icons.OPEN_IN_NEW,
                        icon_size=14,
                        tooltip="Open file location",
                        on_click=lambda e, fp=file_path: self._open_folder(fp),
                    ),
                ], spacing=4, vertical_alignment=CrossAxisAlignment.CENTER),
            )
        
        # Badge row
        badge_items = [
            Container(
                content=Text(type_text, size=10),
                bgcolor=type_color,
                padding=padding.all(5),
                border_radius=border_radius.all(5),
            ),
        ]
        if is_local:
            badge_items.append(
                Container(
                    content=Text("LOCAL", size=10),
                    bgcolor=Colors.TEAL_900,
                    padding=padding.all(5),
                    border_radius=border_radius.all(5),
                ),
            )
        badge_items.extend([
            Container(
                content=Text(info.get("vram", ""), size=10),
                bgcolor=Colors.PURPLE_900,
                padding=padding.all(5),
                border_radius=border_radius.all(5),
            ),
            Container(
                content=Text(status_text, size=10),
                bgcolor=Colors.with_opacity(0.3, status_color),
                padding=padding.all(5),
                border_radius=border_radius.all(5),
            ),
        ])
        info_column_items.append(Row(badge_items, spacing=5))
        
        # Build card content rows
        card_content_items = [
            Row([
                ft.Icon(status_icon, color=status_color, size=28),
                Column(info_column_items, expand=True, spacing=3),
                Column(action_buttons, spacing=5) if action_buttons else Container(),
            ], spacing=15),
        ]
        
        # Add progress bar if downloading
        if is_downloading:
            progress = self._download_progress.get(repo_id, 0.0)
            progress_msg = self._download_status.get(repo_id, "Starting...")
            bar_ctrl = ProgressBar(
                value=progress,
                color=Colors.BLUE_400,
                bgcolor=Colors.GREY_800,
                height=6,
                border_radius=border_radius.all(3),
            )
            text_ctrl = Text(f"{progress*100:.0f}% - {progress_msg}", size=10, color=Colors.BLUE_300)
            card_content_items.append(
                Column([bar_ctrl, text_ctrl], spacing=4),
            )
            # Store references for lightweight progress updates (merge, don't replace)
            ctrl = self._progress_controls.get(repo_id, {})
            ctrl["bar"] = bar_ctrl
            ctrl["text"] = text_ctrl
            ctrl["pct_text"] = pct_text_ctrl
            self._progress_controls[repo_id] = ctrl
        
        card = Card(
            content=Container(
                content=Column(card_content_items, spacing=8),
                padding=padding.all(12),
            ),
        )
        target_list.controls.append(card)
    
    def build(self) -> Container:
        self._load_models()
        self._refresh_downloads_tab()
        
        # Create tabs for image models, chat models, and downloads
        tabs = Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                Tab(
                    text="🎨 Image Models",
                    content=self.image_model_list,
                ),
                Tab(
                    text="💬 Chat / LLM Models",
                    content=self.chat_model_list,
                ),
                Tab(
                    text="📥 Downloads",
                    content=self.downloads_list,
                ),
            ],
            expand=True,
        )
        
        return Container(
            content=Column([
                Row([
                    Text("📦 Model Manager", size=18, weight=FontWeight.BOLD),
                    self.refresh_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Row([
                    Text("🟢 Loaded | 🟡 Downloaded | 🟠 Partial | ⚪ Not Downloaded", size=11, color=Colors.GREY_500),
                    self.status_text,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Container(height=10),
                tabs,
            ]),
            padding=padding.all(15),
            expand=True,
        )


# =============================================================================
# PROMPT LIBRARY TAB
# =============================================================================
class PromptLibraryTab:
    """Manage saved prompts for quick access."""
    
    PROMPTS_FILE = Path(__file__).parent / "config" / "saved_prompts.yaml"
    
    def __init__(self, page: Page, on_use_prompt: Optional[Callable[[str, str], None]] = None, on_switch_tab: Optional[Callable[[str], None]] = None):
        self.page = page
        self.on_use_prompt = on_use_prompt  # Callback when user clicks "Use"
        self.on_switch_tab = on_switch_tab  # Callback to switch tabs
        self.filtered_prompts = []  # Filtered list for search
        self._build_ui()
    
    def _build_ui(self):
        self.prompt_list = ListView(spacing=10, padding=10, expand=True)
        
        # Search field for prompts
        self.search_field = TextField(
            label="🔍 Search prompts",
            on_change=self._on_search_change,
            width=300,
        )
        
        # New prompt form
        self.name_input = TextField(
            label="Prompt Name",
            width=300,
            on_change=self._on_form_change,
        )
        self.prompt_input = TextField(
            label="Prompt Text",
            multiline=True,
            min_lines=6,
            max_lines=12,
            expand=True,
            on_change=self._on_form_change,
        )
        self.negative_input = TextField(
            label="Negative Prompt (optional)",
            multiline=True,
            min_lines=4,
            max_lines=8,
            expand=True,
        )
        
        # Category tags (multiple selection)
        self.category_tags = {
            "General": False,
            "Portrait": False,
            "Landscape": False,
            "Fantasy": False,
            "Sci-Fi": False,
            "Anime": False,
            "Other": False,
        }
        self.category_buttons = {}
        category_row = Row(spacing=5, wrap=True)
        for cat in self.category_tags.keys():
            btn = ElevatedButton(
                cat,
                on_click=lambda e, c=cat: self._toggle_category(c),
                bgcolor=Colors.GREY_700,
                color=Colors.WHITE,
            )
            self.category_buttons[cat] = btn
            category_row.controls.append(btn)
        
        self.save_btn = ElevatedButton(
            "Save Prompt",
            icon=Icons.SAVE,
            on_click=self._save_prompt,
            disabled=True,
        )
        
        self.category_row = category_row
        
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh list",
            on_click=lambda e: self._load_prompts(),
        )
    
    def _toggle_category(self, category: str):
        """Toggle category selection."""
        self.category_tags[category] = not self.category_tags[category]
        btn = self.category_buttons[category]
        if self.category_tags[category]:
            btn.bgcolor = Colors.PRIMARY
            btn.color = Colors.WHITE
        else:
            btn.bgcolor = Colors.GREY_700
            btn.color = Colors.WHITE
        self._update_save_btn_state()
        self.page.update()
    
    def _get_selected_categories(self) -> List[str]:
        """Get list of selected categories."""
        return [cat for cat, selected in self.category_tags.items() if selected]
    
    def _on_search_change(self, e):
        """Handle search field changes."""
        self._filter_and_display_prompts()
    
    def _filter_and_display_prompts(self):
        """Filter prompts based on search text and display."""
        search_text = self.search_field.value.lower().strip()
        data = self._load_saved_prompts()
        prompts = data.get("prompts", [])
        
        if search_text:
            self.filtered_prompts = [
                (idx, p) for idx, p in enumerate(prompts)
                if search_text in p.get("name", "").lower() or 
                   search_text in p.get("prompt", "").lower()
            ]
        else:
            self.filtered_prompts = [(idx, p) for idx, p in enumerate(prompts)]
        
        self._display_prompts()
    
    def _update_save_btn_state(self):
        """Update save button enabled state."""
        has_name = self.name_input.value and self.name_input.value.strip()
        has_prompt = self.prompt_input.value and self.prompt_input.value.strip()
        self.save_btn.disabled = not (has_name and has_prompt)
        try:
            self.page.update()
        except:
            pass
    
    def _on_form_change(self, e):
        """Handle form input changes."""
        self._update_save_btn_state()
    
    def _load_saved_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        if not self.PROMPTS_FILE.exists():
            return {"prompts": []}
        
        try:
            import yaml
            with open(self.PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {"prompts": []}
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            return {"prompts": []}
    
    def _save_prompts_file(self, data: Dict[str, Any]) -> bool:
        """Save prompts to YAML file."""
        try:
            import yaml
            with open(self.PROMPTS_FILE, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
            return False
    
    def _load_prompts(self):
        """Load and display saved prompts (with search support)."""
        self.search_field.value = ""
        self._filter_and_display_prompts()
    
    def _display_prompts(self):
        """Display the filtered/searched prompts."""
        self.prompt_list.controls.clear()
        
        if not self.filtered_prompts:
            self.prompt_list.controls.append(
                Container(
                    content=Text("No prompts found. Add one below!", 
                                color=Colors.GREY_500, italic=True),
                    padding=padding.all(20),
                )
            )
            self.page.update()
            return
        
        for idx_in_all, prompt in self.filtered_prompts:
            name = prompt.get("name", "Untitled")
            text = prompt.get("prompt", "")
            negative = prompt.get("negative", "")
            categories = prompt.get("categories", [prompt.get("category", "general")])  # Support both old and new format
            
            # Category badge colors
            cat_colors = {
                "portrait": Colors.PURPLE_700,
                "landscape": Colors.GREEN_700,
                "fantasy": Colors.ORANGE_700,
                "sci-fi": Colors.BLUE_700,
                "anime": Colors.PINK_700,
                "general": Colors.GREY_700,
                "other": Colors.BROWN_700,
            }
            
            # Create category tags
            category_tags = Row(spacing=5, wrap=True)
            if isinstance(categories, list):
                for cat in categories:
                    category_tags.controls.append(
                        Container(
                            content=Text(cat.capitalize(), size=9),
                            bgcolor=cat_colors.get(cat.lower(), Colors.GREY_700),
                            padding=padding.symmetric(horizontal=6, vertical=2),
                            border_radius=border_radius.all(6),
                        )
                    )
            else:
                category_tags.controls.append(
                    Container(
                        content=Text(categories.capitalize(), size=9),
                        bgcolor=cat_colors.get(categories.lower(), Colors.GREY_700),
                        padding=padding.symmetric(horizontal=6, vertical=2),
                        border_radius=border_radius.all(6),
                    )
                )
            
            card = Card(
                content=Container(
                    content=Column([
                        Row([
                            Text(name, weight=FontWeight.BOLD, size=14, expand=True),
                        ]),
                        category_tags,
                        Text(
                            text[:100] + "..." if len(text) > 100 else text,
                            size=11, color=Colors.GREY_400,
                        ),
                        Container(
                            content=Text(
                                f"🚫 {negative[:80] + '...' if len(negative) > 80 else negative}",
                                size=10, color=Colors.ORANGE_300, italic=True,
                            ) if negative else Text("", size=0),
                        ),
                        Row([
                            ElevatedButton(
                                "Use",
                                icon=Icons.PLAY_ARROW,
                                on_click=lambda e, p=text, n=negative: self._use_prompt(p, n),
                            ),
                            IconButton(
                                icon=Icons.CONTENT_COPY,
                                tooltip="Copy prompt",
                                on_click=lambda e, p=text: self._copy_prompt(p),
                            ),
                            IconButton(
                                icon=Icons.FILE_COPY,
                                tooltip="Clone",
                                icon_color=Colors.BLUE_400,
                                on_click=lambda e, i=idx_in_all: self._clone_prompt(i),
                            ),
                            IconButton(
                                icon=Icons.DELETE_OUTLINE,
                                tooltip="Delete",
                                icon_color=Colors.RED_400,
                                on_click=lambda e, i=idx_in_all: self._delete_prompt(i),
                            ),
                        ], spacing=5),
                    ], spacing=6),
                    padding=padding.all(12),
                ),
            )
            self.prompt_list.controls.append(card)
        
        self.page.update()
    
    def _use_prompt(self, prompt: str, negative: str):
        """Use a saved prompt in the image generator."""
        if self.on_use_prompt:
            self.on_use_prompt(prompt, negative)
        
        # Switch to Generate tab
        if self.on_switch_tab:
            self.on_switch_tab(0)  # 0 = ImageGenTab
        
        # Show snackbar
        self.page.snack_bar = SnackBar(
            content=Text("✅ Prompt loaded! Switched to Generate tab."),
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def _clone_prompt(self, index: int):
        """Clone a saved prompt."""
        data = self._load_saved_prompts()
        if 0 <= index < len(data["prompts"]):
            original = data["prompts"][index]
            clone = original.copy()
            clone["name"] = f"{clone['name']} (Copy)"
            data["prompts"].append(clone)
            
            if self._save_prompts_file(data):
                self.page.snack_bar = SnackBar(content=Text(f"✂️ Cloned: {clone['name']}"))
                self.page.snack_bar.open = True
                self._filter_and_display_prompts()
            else:
                self.page.snack_bar = SnackBar(
                    content=Text("Failed to clone prompt."),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
                self.page.update()
    
    def _copy_prompt(self, prompt: str):
        """Copy prompt to clipboard."""
        self.page.set_clipboard(prompt)
        self.page.snack_bar = SnackBar(content=Text("Prompt copied!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _save_prompt(self, e):
        """Save a new prompt."""
        name = self.name_input.value.strip()
        prompt = self.prompt_input.value.strip()
        negative = self.negative_input.value.strip()
        categories = self._get_selected_categories()
        
        # Default to "General" if no categories selected
        if not categories:
            categories = ["General"]
        
        data = self._load_saved_prompts()
        data["prompts"].append({
            "name": name,
            "prompt": prompt,
            "negative": negative,
            "categories": categories,
        })
        
        if self._save_prompts_file(data):
            # Clear form
            self.name_input.value = ""
            self.prompt_input.value = ""
            self.negative_input.value = ""
            
            # Reset category buttons
            for cat in self.category_tags:
                self.category_tags[cat] = False
                self.category_buttons[cat].bgcolor = Colors.GREY_700
            
            self._update_save_btn_state()
            self.page.snack_bar = SnackBar(content=Text(f"✅ Saved: {name}"))
            self.page.snack_bar.open = True
            self._load_prompts()
        else:
            self.page.snack_bar = SnackBar(
                content=Text("Failed to save prompt."),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _delete_prompt(self, index: int):
        """Delete a saved prompt."""
        data = self._load_saved_prompts()
        if 0 <= index < len(data["prompts"]):
            deleted = data["prompts"].pop(index)
            self._save_prompts_file(data)
            self.page.snack_bar = SnackBar(content=Text(f"Deleted: {deleted.get('name', 'prompt')}"))
            self.page.snack_bar.open = True
            self._load_prompts()
    
    def add_prompt_from_generator(self, prompt: str, negative: str, settings: dict = None):
        """Add a prompt from the image generator (called externally)."""
        from datetime import datetime
        
        # Generate a name based on first few words
        words = prompt.split()[:5]
        name = " ".join(words) + ("..." if len(words) >= 5 else "")
        name = f"{name} ({datetime.now().strftime('%H:%M')})"
        
        data = self._load_saved_prompts()
        prompt_data = {
            "name": name,
            "prompt": prompt,
            "negative": negative,
            "category": "general",
        }
        
        # Store settings if provided
        if settings:
            prompt_data["settings"] = settings
        
        data["prompts"].append(prompt_data)
        
        if self._save_prompts_file(data):
            self.page.snack_bar = SnackBar(content=Text(f"💾 Saved to library: {name[:30]}..."))
            self.page.snack_bar.open = True
            self._load_prompts()
        else:
            self.page.snack_bar = SnackBar(
                content=Text("Failed to save prompt."),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def build(self) -> Container:
        self._load_prompts()
        
        return Container(
            content=Row([
                # Saved prompts list
                Container(
                    content=Column([
                        Row([
                            Text("📚 Saved Prompts", size=18, weight=FontWeight.BOLD),
                            self.refresh_btn,
                        ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                        self.search_field,
                        self.prompt_list,
                    ], spacing=10),
                    expand=2,
                    padding=padding.all(15),
                ),
                # Add new prompt form
                Container(
                    content=Column([
                        Text("➕ Add New Prompt", size=16, weight=FontWeight.BOLD),
                        Container(height=5),
                        self.name_input,
                        Text("Categories (select one or more):", size=12, color=Colors.GREY_400),
                        self.category_row,
                        Text("Prompt:", size=12, color=Colors.GREY_400),
                        self.prompt_input,
                        Text("Negative Prompt:", size=12, color=Colors.GREY_400),
                        self.negative_input,
                        self.save_btn,
                    ], spacing=8, expand=True),
                    width=400,
                    padding=padding.all(15),
                    bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
                    border_radius=border_radius.all(10),
                ),
            ], expand=True),
            expand=True,
        )


# =============================================================================
# CHAT TAB
# =============================================================================
class ChatTab:
    """Chat interface with character/persona management - uses the model loaded from header."""
    
    def __init__(self, page: Page):
        self.page = page
        self.db = ChatHistoryDB()  # uses core.get_effective_db_path(); see Settings > Database Location
        self.selected_character = None  # Current selected character dict
        self.selected_persona = None  # Current selected persona dict
        self.editing_character_id = None  # ID of character being edited (None = new)
        self.editing_persona_id = None  # ID of persona being edited (None = new)
        # Conversation tracking
        self.current_conversation_id = None  # Active conversation DB id
        self.last_user_message = None  # For regeneration
        # Regen carousel state
        self.regen_responses = []  # All generated responses for current turn
        self.regen_index = 0  # Currently displayed response index
        self.response_committed = True  # Whether current response is committed to history
        self.regen_text_ctrl = None  # Text widget in carousel bubble
        self.regen_counter_text = None  # "1/3" counter text widget
        self.regen_prev_btn = None  # Prev button
        self.regen_next_btn = None  # Next button
        self.directions_visible = False  # Whether directions input is expanded
        self._build_ui()
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _build_ui(self):
        """Build all UI components for the chat tab with sub-tabs."""
        # =========== CONVERSATION TAB COMPONENTS ===========
        
        # Model status (shows what's loaded from header)
        self.model_status = Text("💡 Load a 💬 Chat model from the header dropdown", size=12, color=Colors.GREY_500)
        
        # Character dropdown
        self.char_dropdown = Dropdown(
            label="AI Character",
            width=240,
            options=self._get_character_options(),
            value="none",
            on_change=self._on_character_change,
        )
        
        # Persona dropdown
        self.persona_dropdown = Dropdown(
            label="User Persona",
            width=240,
            options=self._get_persona_options(),
            value="none",
            on_change=self._on_persona_change,
        )
        
        # System prompt (hidden when character is selected)
        self.system_prompt = TextField(
            label="System Prompt",
            value="You are a helpful AI assistant.",
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        
        # Generation parameters with labels
        self.temp_value_text = Text("", size=10, width=55)
        self.temp_slider = Slider(
            min=0.1, max=1.5, value=0.7,
            divisions=14,
            width=130,
            on_change=lambda e: self._update_slider_label(self.temp_value_text, "Temp {:.1f}", self.temp_slider.value),
        )
        
        self.max_tokens_value_text = Text("", size=10, width=65)
        self.max_tokens_slider = Slider(
            min=64, max=512, value=256,
            divisions=7,
            width=130,
            on_change=lambda e: self._update_slider_label(self.max_tokens_value_text, "{} tokens", int(self.max_tokens_slider.value)),
        )
        
        # Initialize Chat tab slider labels
        self._update_slider_label(self.temp_value_text, "Temp {:.1f}", self.temp_slider.value)
        self._update_slider_label(self.max_tokens_value_text, "{} tokens", int(self.max_tokens_slider.value))
        
        # Chat messages list
        self.chat_messages = ListView(
            spacing=10,
            padding=padding.all(10),
            expand=True,
            auto_scroll=True,
        )
        
        # Message input with Enter to send (shift_enter makes Enter submit)
        self.message_input = TextField(
            label="Type your message... (use *actions* for narration, Enter to send, Shift+Enter for newline)",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
            shift_enter=True,
            on_submit=self._send_message,
            on_change=self._on_message_input_change,
        )
        
        # Directions input (hidden by default)
        self.directions_input = TextField(
            label="Optional: Give directions on how character should respond (not sent as dialog)",
            hint_text="e.g. 'respond with confusion and fear' or 'stand up and look around'",
            multiline=True,
            min_lines=2,
            max_lines=3,
            expand=True,
            visible=False,
        )
        
        # Toggle button to expand/collapse directions input
        self.directions_toggle = IconButton(
            icon=Icons.EXPAND_MORE,
            tooltip="Show directions (optional instructions for character)",
            on_click=lambda e: self._toggle_directions(),
            icon_size=20,
        )
        
        self.send_btn = ElevatedButton(
            "Send",
            icon=Icons.SEND,
            on_click=self._send_message,
            disabled=True,  # Disabled until text is entered
            style=ButtonStyle(
                bgcolor={
                    ControlState.DEFAULT: Colors.BLUE_700,
                    ControlState.DISABLED: Colors.GREY_700,
                },
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
                overlay_color=Colors.TRANSPARENT,
            ),
        )
        
        # ===== DEBUG / PROMPT CONSOLE =====
        # Each section is built dynamically in _populate_debug_console
        self.debug_panel_list = Column([], spacing=0)
        self.debug_copy_btn = IconButton(
            icon=Icons.COPY,
            tooltip="Copy all to clipboard",
            on_click=self._copy_debug_to_clipboard,
            icon_size=16,
        )
        self.debug_no_data_text = Text(
            "Send a message to see the prompt debug info here.",
            size=13, color=Colors.GREY_500, italic=True,
        )
        self.debug_color_key = Row([
            Container(width=12, height=12, bgcolor=Colors.PINK_200, border_radius=2),
            Text("System", size=9, color=Colors.PINK_200),
            Container(width=8),
            Container(width=12, height=12, bgcolor=Colors.LIGHT_BLUE_400, border_radius=2),
            Text("User", size=9, color=Colors.LIGHT_BLUE_400),
            Container(width=8),
            Container(width=12, height=12, bgcolor=Colors.GREEN_200, border_radius=2),
            Text("LLM", size=9, color=Colors.GREEN_200),
            Container(width=8),
            Container(width=12, height=12, bgcolor=Colors.AMBER_400, border_radius=2),
            Text("Tags", size=9, color=Colors.AMBER_400),
        ], spacing=4, visible=False)
        self.debug_console_content = Container(
            content=Column([
                Row([
                    Text("🔍 Prompt Debug Console", size=16, weight=FontWeight.BOLD, color=Colors.AMBER_300),
                    self.debug_copy_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                self.debug_color_key,
                self.debug_no_data_text,
                Container(
                    content=self.debug_panel_list,
                    expand=True,
                ),
            ], spacing=6, expand=True, scroll=ScrollMode.AUTO),
            padding=padding.all(15),
            expand=True,
        )
        
        self.clear_btn = IconButton(
            icon=Icons.DELETE_SWEEP,
            tooltip="Delete conversation",
            on_click=self._delete_current_conversation,
        )
        
        # Memory button (shows count badge when memories exist)
        self.memory_btn = ElevatedButton(
            "🧠 Memories",
            on_click=self._show_memories_dialog,
            height=30,
            style=ButtonStyle(
                padding=padding.symmetric(horizontal=8, vertical=2),
                text_style=ft.TextStyle(size=11),
            ),
        )
        
        # Typing indicator (positioned above message input)
        self.typing_indicator = Row([
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=Colors.GREEN_400),
            Text("AI is typing...", size=11, color=Colors.GREEN_400, italic=True),
        ], spacing=8, visible=False)
        
        # No model banner
        self.no_model_banner = Container(
            content=Row([
                ft.Icon(Icons.WARNING_AMBER_ROUNDED, size=24, color=Colors.AMBER_400),
                Text("💬 No chat model loaded! Load a model from the header dropdown to chat.", 
                     size=14, color=Colors.AMBER_300),
            ], spacing=10, alignment=MainAxisAlignment.CENTER),
            bgcolor=Colors.with_opacity(0.3, Colors.AMBER_900),
            padding=padding.all(12),
            border_radius=border_radius.all(8),
            visible=True,  # Show by default until model loads
        )
        
        # Progress indicator
        self.progress = ProgressBar(visible=False)
        self.generating = False
        
        # File picker for export (will be added to page overlay in build())
        self.export_file_picker = ft.FilePicker(on_result=self._on_export_file_picked)
        self._pending_export_data = None  # Holds data to export when file picker returns
        
        # Suggest response button (shown after first AI response)
        self.suggest_prompt_btn = ElevatedButton(
            "💡 Suggest Response",
            on_click=self._suggest_prompt,
            visible=False,
            height=30,
            style=ButtonStyle(
                padding=padding.symmetric(horizontal=10, vertical=2),
                text_style=ft.TextStyle(size=11),
            ),
        )
        self.suggested_prompt_container = Container(
            content=Column([], spacing=4),
            visible=False,
            bgcolor=Colors.with_opacity(0.2, Colors.BLUE_900),
            padding=padding.all(8),
            border_radius=border_radius.all(8),
        )
        # Suggested response carousel state
        self.suggestion_responses = []  # All suggested responses
        self.suggestion_index = 0  # Currently displayed suggestion index
        self.suggestion_text_ctrl = None  # Text widget in suggestion display
        self.suggestion_counter_text = None  # "1/3" counter
        self.suggestion_prev_btn = None  # Prev button
        self.suggestion_next_btn = None  # Next button
        
        # =========== HISTORY TAB COMPONENTS ===========
        self.history_list = ListView(
            spacing=8,
            padding=padding.only(top=4, bottom=4, left=10, right=10),
            expand=True,
        )
        self.history_search = TextField(
            label="🔍 Search conversations...",
            on_change=self._refresh_history_list,
        )
        self.history_delete_all_btn = ElevatedButton(
            "🗑️ Delete All",
            on_click=self._confirm_delete_all_conversations,
            color=Colors.RED_400,
            height=30,
            style=ButtonStyle(
                padding=padding.symmetric(horizontal=10, vertical=2),
                text_style=ft.TextStyle(size=11),
            ),
        )
        self.history_status = Text("", size=12)
        self.history_export_dialog_content = Text("", selectable=True)
        
        # =========== CHARACTERS TAB COMPONENTS ===========
        
        # Character list for browsing
        self.char_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Character edit form
        self.char_name_input = TextField(label="Character Name", hint_text="e.g., Friendly Tutor")
        self.char_desc_input = TextField(label="Short Description", hint_text="Brief dropdown description")
        self.char_system_input = TextField(
            label="System Prompt (Personality)",
            hint_text="You are a helpful tutor who explains things clearly...",
            multiline=True,
            min_lines=5,
            max_lines=10,
        )
        self.char_avatar_input = TextField(label="Avatar Emoji", value="🤖", width=100)
        
        # Character editor sliders with labels
        self.char_temp_value_text = Text("", size=11)
        self.char_temp_slider = Slider(min=0.1, max=1.5, value=0.7, divisions=14,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_temp_value_text, "Temp {:.1f}", self.char_temp_slider.value))
        
        self.char_topp_value_text = Text("", size=11)
        self.char_topp_slider = Slider(min=0.1, max=1.0, value=0.95, divisions=18,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_topp_value_text, "Top-P {:.2f}", self.char_topp_slider.value))
        
        self.char_topk_value_text = Text("", size=11)
        self.char_topk_slider = Slider(min=1, max=100, value=50, divisions=99,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_topk_value_text, "Top-K {}", int(self.char_topk_slider.value)))
        
        # Initialize character slider labels
        self._update_slider_label(self.char_temp_value_text, "Temp {:.1f}", self.char_temp_slider.value)
        self._update_slider_label(self.char_topp_value_text, "Top-P {:.2f}", self.char_topp_slider.value)
        self._update_slider_label(self.char_topk_value_text, "Top-K {}", int(self.char_topk_slider.value))
        
        self.char_save_btn = ElevatedButton("💾 Save Character", icon=Icons.SAVE, on_click=self._save_character)
        self.char_delete_btn = ElevatedButton("🗑️ Delete", icon=Icons.DELETE, on_click=self._delete_character, color=Colors.RED_400)
        self.char_back_btn = TextButton("← Back to Gallery", on_click=self._show_char_browse)
        
        # Character prompt override controls (shown inside expansion panel)
        self.char_override_mode = Dropdown(
            label="Override Mode",
            width=180,
            options=[
                dropdown.Option("replace", "Replace — use this instead of default"),
                dropdown.Option("append", "Append — add after default"),
            ],
            value="replace",
        )
        self.char_override_char_instructions = TextField(
            label="Character Instructions Override",
            hint_text="Leave blank to use global default. No placeholders needed.",
            multiline=True, min_lines=2, max_lines=5,
        )
        self.char_override_memory = TextField(
            label="Memory Template Override",
            hint_text="Leave blank to use global default. Supports {memories}.",
            multiline=True, min_lines=2, max_lines=5,
        )
        
        # Browse vs Edit views
        self.char_browse_view = Column([], expand=True)
        self.char_edit_view = Column([], expand=True, visible=False, scroll=ScrollMode.AUTO)
        
        self.char_status = Text("", size=12, color=Colors.GREEN_400)
        
        # =========== PERSONAS TAB COMPONENTS ===========
        
        # Persona list for browsing
        self.persona_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Persona edit form
        self.persona_name_input = TextField(label="Persona Name", hint_text="e.g., Space Explorer")
        self.persona_desc_input = TextField(label="Short Description", hint_text="Brief dropdown description")
        self.persona_bg_input = TextField(
            label="Background/Context",
            hint_text="Detailed roleplay background: You are Captain Sarah Chen...",
            multiline=True,
            min_lines=5,
            max_lines=10,
        )
        self.persona_avatar_input = TextField(label="Avatar Emoji", value="👤", width=100)
        
        self.persona_save_btn = ElevatedButton("💾 Save Persona", icon=Icons.SAVE, on_click=self._save_persona)
        self.persona_delete_btn = ElevatedButton("🗑️ Delete", icon=Icons.DELETE, on_click=self._delete_persona, color=Colors.RED_400)
        self.persona_back_btn = TextButton("← Back to Gallery", on_click=self._show_persona_browse)
        
        # Browse vs Edit views
        self.persona_browse_view = Column([], expand=True)
        self.persona_edit_view = Column([], expand=True, visible=False)
        
        self.persona_status = Text("", size=12, color=Colors.GREEN_400)
        
        # =========== PROMPT TEMPLATES TAB COMPONENTS ===========
        
        self.pt_char_instructions_input = TextField(
            label="Character Instructions (always sent)",
            hint_text="Baseline rules for all characters. No placeholders needed.",
            multiline=True, min_lines=4, max_lines=10, expand=True,
        )
        self.pt_persona_input = TextField(
            label="Persona Context Template",
            hint_text="Use {persona_name} and {persona_background} as placeholders",
            multiline=True, min_lines=4, max_lines=10, expand=True,
        )
        self.pt_directions_input = TextField(
            label="Directions Template",
            hint_text="Use {directions} as placeholder",
            multiline=True, min_lines=3, max_lines=8, expand=True,
        )
        self.pt_memory_input = TextField(
            label="Memory Template",
            hint_text="Use {memories} as placeholder",
            multiline=True, min_lines=3, max_lines=8, expand=True,
        )
        self.pt_stop_phrases_input = TextField(
            label="Stop Phrases (one per line)",
            hint_text="\\nUser:\\n\\nAssistant:\\netc.",
            multiline=True, min_lines=3, max_lines=8, expand=True,
        )
        self.pt_char_name_stop = Checkbox(label="Auto-add character name as stop phrase", value=True)
        self.pt_persona_name_stop = Checkbox(label="Auto-add persona name as stop phrase", value=True)
        self.pt_save_btn = ElevatedButton("💾 Save Templates", icon=Icons.SAVE, on_click=self._save_prompt_templates)
        self.pt_reset_btn = TextButton("↺ Reset to Defaults", on_click=self._reset_prompt_templates)
        self.pt_status = Text("", size=12, color=Colors.GREEN_400)
    
    def _get_character_options(self):
        """Get character options for dropdown."""
        options = [dropdown.Option("none", "None (Use System Prompt)")]
        for char in self.db.get_all_ai_characters():
            options.append(dropdown.Option(
                str(char['id']),
                f"{char['avatar']} {char['name']}"
            ))
        return options
    
    def _get_persona_options(self):
        """Get persona options for dropdown."""
        options = [dropdown.Option("none", "None")]
        for persona in self.db.get_all_user_personas():
            options.append(dropdown.Option(
                str(persona['id']),
                f"{persona['avatar']} {persona['name']}"
            ))
        return options
    
    def _refresh_dropdowns(self):
        """Refresh character and persona dropdowns."""
        self.char_dropdown.options = self._get_character_options()
        self.persona_dropdown.options = self._get_persona_options()
    
    def _on_character_change(self, e):
        """Handle character selection."""
        if e.control.value == "none":
            self.selected_character = None
            self.system_prompt.visible = True
            self.temp_slider.value = 0.7
        else:
            char_id = int(e.control.value)
            self.selected_character = self.db.get_ai_character(char_id)
            if self.selected_character:
                self.system_prompt.visible = False
                self.temp_slider.value = self.selected_character.get('temperature', 0.7)
        self.page.update()
    
    def _on_persona_change(self, e):
        """Handle persona selection."""
        if e.control.value == "none":
            self.selected_persona = None
        else:
            persona_id = int(e.control.value)
            self.selected_persona = self.db.get_user_persona(persona_id)
        self.page.update()
    
    def update_model_state(self):
        """Update UI based on current model state."""
        has_model = app_state.loaded_model_type == "chat"
        has_text = bool(self.message_input.value and self.message_input.value.strip())
        self.send_btn.disabled = not (has_model and has_text)
        self.no_model_banner.visible = not has_model
        try:
            self.page.update()
        except:
            pass
    
    def _on_message_input_change(self, e):
        """Enable/disable send button based on message input content."""
        has_text = bool(self.message_input.value and self.message_input.value.strip())
        has_model = app_state.loaded_model_type == "chat"
        should_enable = has_text and has_model and not self.generating
        if self.send_btn.disabled != (not should_enable):
            self.send_btn.disabled = not should_enable
            try:
                self.page.update()
            except:
                pass
    
    def _toggle_directions(self):
        """Toggle visibility of the directions input area."""
        self.directions_visible = not self.directions_visible
        self.directions_input.visible = self.directions_visible
        self.directions_toggle.icon = (
            Icons.EXPAND_LESS if self.directions_visible else Icons.EXPAND_MORE
        )
        self.directions_toggle.tooltip = (
            "Hide directions" if self.directions_visible
            else "Show directions (optional instructions for character)"
        )
        self.page.update()
    
    def _switch_to_debug_tab(self):
        """Switch the chat sub-tabs to the Debug Console tab."""
        if hasattr(self, '_chat_sub_tabs'):
            # Debug tab is the last tab
            self._chat_sub_tabs.selected_index = len(self._chat_sub_tabs.tabs) - 1
            try:
                self.page.update()
            except:
                pass

    def _copy_debug_to_clipboard(self, e=None):
        """Copy the debug console text to clipboard as plain text."""
        from core.chat_gen import _last_debug_info as dbg
        if not dbg:
            return
        # Build a plain-text version for clipboard
        sections = []
        # Prompt components
        sections.append(("── PROMPT COMPONENTS ──", ""))
        sections.append(("BASE SYSTEM PROMPT", dbg.get("_base_system", "")))
        if dbg.get("_char_instructions"):
            sections.append(("CHARACTER INSTRUCTIONS (always sent)", dbg["_char_instructions"]))
        if dbg.get("_persona_name"):
            sections.append(("PERSONA", f"{dbg['_persona_name']}: {dbg.get('_persona_bg', '')}"))
        # Scene Instructions as separate section
        directions_text = dbg.get("_directions", "").strip()
        if directions_text:
            sections.append(("── SCENE INSTRUCTIONS ──", ""))
            sections.append(("INSTRUCTIONS (this turn)", directions_text))
        # Memories
        mem_list = dbg.get("_memories", [])
        if mem_list:
            retrieval = dbg.get("_retrieval")
            sections.append(("── MEMORIES ──", ""))
            if retrieval:
                mem_lines = []
                for c in retrieval.selected:
                    pin = " 📌" if c.pinned else ""
                    mem_lines.append(f"  {c.score:+.3f}  {c.text}{pin}")
                sections.append(("SELECTED MEMORIES", "\n".join(mem_lines)))
            else:
                sections.append(("MEMORIES (all injected)", "\n".join(f"  • {m}" for m in mem_list)))
        # Conversation history
        history_turns = dbg.get("history_turns", [])
        if history_turns:
            sections.append(("── CONVERSATION HISTORY ──", ""))
            hist_lines = []
            turn_num = 0
            for t in history_turns:
                role = t.get("role", "unknown")
                content = t.get("content", "")
                if role == "user":
                    turn_num += 1
                    hist_lines.append(f"── Turn {turn_num} ──")
                    hist_lines.append(f"  User: {content}")
                elif role == "assistant":
                    if len(content) > 300:
                        content = content[:300] + "..."
                    hist_lines.append(f"  Assistant: {content}")
            sections.append(("PRIOR MESSAGES", "\n".join(hist_lines)))
        # Assembled
        sections.append(("── ASSEMBLED & SENT ──", ""))
        sections.append(("ASSEMBLED SYSTEM PROMPT", dbg.get("system_prompt", "")))
        sections.append(("USER MESSAGE", dbg.get("user_message", "")))
        sections.append(("HISTORY TURNS", str(dbg.get("history_length", 0))))
        sections.append(("TOKENS", f"input={dbg.get('input_tokens','?')}, output={dbg.get('output_tokens','?')}"))
        # Pipeline
        sections.append(("── GENERATION PIPELINE ──", ""))
        sections.append(("FULL FORMATTED PROMPT", dbg.get("full_prompt", "")))
        sections.append(("RAW RESPONSE", dbg.get("raw_response", "")))
        sections.append(("STOP PHRASES", ", ".join(repr(s) for s in dbg.get("stop_phrases", []))))
        sections.append(("CLEANED RESPONSE", dbg.get("cleaned_response", "")))
        if "error" in dbg:
            sections.append(("ERROR", dbg["error"]))
        lines = ["=" * 60, "  PROMPT DEBUG — Last Generation", "=" * 60]
        for title, body in sections:
            if body:
                lines.append(f"\n── {title} {'─' * max(1, 45 - len(title))}")
                lines.append(body)
            elif title.startswith("──"):
                lines.append(f"\n{title}")
        lines.append("\n" + "=" * 60)
        self.page.set_clipboard("\n".join(lines))
        self.page.snack_bar = SnackBar(content=Text("Debug info copied to clipboard"))
        self.page.snack_bar.open = True
        self.page.update()

    # ── Rich-text span builders for debug console ──────────────

    def _build_rich_body_spans(self, body: str, text_color: str) -> list:
        """Parse bracket tags [LIKE THIS] and color them distinctly from body text."""
        import re
        TAG_COLOR = Colors.AMBER_400
        parts = re.split(r'(\[/?[A-Z][A-Z _/]*\])', body)
        spans = []
        for part in parts:
            if not part:
                continue
            if re.match(r'^\[/?[A-Z][A-Z _/]*\]$', part):
                spans.append(ft.TextSpan(
                    text=part,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=TAG_COLOR, weight=FontWeight.BOLD),
                ))
            else:
                spans.append(ft.TextSpan(
                    text=part,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=text_color),
                ))
        return spans

    def _build_history_spans(self, body: str) -> list:
        """Build rich text spans for conversation history with role coloring."""
        HEADER_COLOR = Colors.GREY_500
        USER_COLOR = Colors.LIGHT_BLUE_400
        ASST_COLOR = Colors.GREEN_200
        DEFAULT_COLOR = Colors.GREY_300
        spans = []
        for line in body.split('\n'):
            if spans:
                spans.append(ft.TextSpan(
                    text='\n',
                    style=ft.TextStyle(size=11, font_family="Courier New"),
                ))
            stripped = line.strip()
            if stripped.startswith('── Turn'):
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=HEADER_COLOR, weight=FontWeight.BOLD),
                ))
            elif '  User: ' in line or stripped.startswith('User: '):
                idx = line.index('User:')
                prefix = line[:idx]
                label = 'User:'
                content = line[idx + 5:]
                if prefix:
                    spans.append(ft.TextSpan(text=prefix,
                        style=ft.TextStyle(size=11, font_family="Courier New", color=DEFAULT_COLOR)))
                spans.append(ft.TextSpan(text=label,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=USER_COLOR, weight=FontWeight.BOLD)))
                spans.append(ft.TextSpan(text=content,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=USER_COLOR)))
            elif '  Assistant: ' in line or stripped.startswith('Assistant: '):
                idx = line.index('Assistant:')
                prefix = line[:idx]
                label = 'Assistant:'
                content = line[idx + 10:]
                if prefix:
                    spans.append(ft.TextSpan(text=prefix,
                        style=ft.TextStyle(size=11, font_family="Courier New", color=DEFAULT_COLOR)))
                spans.append(ft.TextSpan(text=label,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=ASST_COLOR, weight=FontWeight.BOLD)))
                spans.append(ft.TextSpan(text=content,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=ASST_COLOR)))
            else:
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=DEFAULT_COLOR),
                ))
        return spans

    def _build_memory_spans(self, body: str) -> list:
        """Build rich text spans for memory display with score coloring."""
        import re
        HEADER_COLOR = Colors.YELLOW_200
        SCORE_HIGH = Colors.GREEN_300
        SCORE_MED = Colors.YELLOW_300
        SCORE_LOW = Colors.ORANGE_300
        PIN_COLOR = Colors.AMBER_300
        LABEL_COLOR = Colors.GREY_400
        TEXT_COLOR = Colors.GREY_300
        spans = []
        for line in body.split('\n'):
            if spans:
                spans.append(ft.TextSpan(
                    text='\n',
                    style=ft.TextStyle(size=11, font_family="Courier New"),
                ))
            stripped = line.strip()
            if stripped.startswith('── ') and stripped.endswith(' ──'):
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=HEADER_COLOR, weight=FontWeight.BOLD),
                ))
            elif re.match(r'^\s*[+-]\d+\.\d+', stripped):
                match = re.match(r'^(\s*)([+-]\d+\.\d+)(\s+)(.*?)(\s*📌)?$', line)
                if match:
                    indent, score, gap, text, pin = match.groups()
                    score_val = float(score)
                    if score_val >= 0.5:
                        score_color = SCORE_HIGH
                    elif score_val >= 0.3:
                        score_color = SCORE_MED
                    else:
                        score_color = SCORE_LOW
                    if indent:
                        spans.append(ft.TextSpan(text=indent,
                            style=ft.TextStyle(size=11, font_family="Courier New")))
                    spans.append(ft.TextSpan(text=score,
                        style=ft.TextStyle(size=11, font_family="Courier New",
                                           color=score_color, weight=FontWeight.BOLD)))
                    spans.append(ft.TextSpan(text=gap,
                        style=ft.TextStyle(size=11, font_family="Courier New")))
                    spans.append(ft.TextSpan(text=text,
                        style=ft.TextStyle(size=11, font_family="Courier New", color=TEXT_COLOR)))
                    if pin:
                        spans.append(ft.TextSpan(text=pin,
                            style=ft.TextStyle(size=11, font_family="Courier New", color=PIN_COLOR)))
                else:
                    spans.append(ft.TextSpan(text=line,
                        style=ft.TextStyle(size=11, font_family="Courier New", color=TEXT_COLOR)))
            elif stripped.startswith('Query:') or stripped.startswith('Pool:') or stripped.startswith('Token budget:'):
                colon_idx = stripped.index(':')
                label = stripped[:colon_idx + 1]
                rest = stripped[colon_idx + 1:]
                indent = line[:len(line) - len(line.lstrip())]
                if indent:
                    spans.append(ft.TextSpan(text=indent,
                        style=ft.TextStyle(size=11, font_family="Courier New")))
                spans.append(ft.TextSpan(text=label,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=LABEL_COLOR, weight=FontWeight.BOLD)))
                spans.append(ft.TextSpan(text=rest,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=TEXT_COLOR)))
            elif stripped.startswith('•'):
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=TEXT_COLOR),
                ))
            else:
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=TEXT_COLOR),
                ))
        return spans

    def _build_stop_phrase_spans(self, body: str) -> list:
        """Build rich spans for stop phrases — each repr'd phrase gets highlighted."""
        PHRASE_COLOR = Colors.ORANGE_200
        REPR_COLOR = Colors.ORANGE_400
        spans = []
        for line in body.split('\n'):
            if spans:
                spans.append(ft.TextSpan(
                    text='\n',
                    style=ft.TextStyle(size=11, font_family="Courier New"),
                ))
            stripped = line.strip()
            if stripped.startswith("'") or stripped.startswith('"'):
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=REPR_COLOR),
                ))
            else:
                spans.append(ft.TextSpan(
                    text=line,
                    style=ft.TextStyle(size=11, font_family="Courier New",
                                       color=PHRASE_COLOR),
                ))
        return spans

    # ── Debug section builders ───────────────────────────────

    def _build_debug_section(self, title: str, body: str, color: str, icon: str,
                              initially_expanded: bool = False,
                              custom_spans: list = None) -> ft.ExpansionPanelList:
        """Build a single color-coded collapsible debug section with rich text and copy button."""
        if custom_spans is not None:
            spans = custom_spans
        else:
            spans = self._build_rich_body_spans(body, color)

        text_widget = ft.SelectionArea(
            content=ft.Text(
                spans=spans,
                selectable=True,
            ),
        )

        # Per-section copy button (captures body in closure)
        plain_body = body
        def _copy_this(e, _text=plain_body, _title=title):
            self.page.set_clipboard(_text)
            self.page.snack_bar = SnackBar(content=Text(f"Copied: {_title}"))
            self.page.snack_bar.open = True
            self.page.update()

        copy_btn = IconButton(
            icon=Icons.COPY,
            icon_size=14,
            tooltip=f"Copy {title}",
            on_click=_copy_this,
            style=ButtonStyle(padding=padding.all(0)),
        )

        return ft.ExpansionPanelList(
            elevation=0,
            expanded_header_padding=padding.all(0),
            controls=[
                ft.ExpansionPanel(
                    header=ft.ListTile(
                        leading=Text(icon, size=14),
                        title=Row([
                            Text(title, size=12, weight=FontWeight.BOLD, color=color),
                            copy_btn,
                        ], spacing=4, alignment=MainAxisAlignment.START),
                        dense=True,
                        content_padding=padding.only(left=8),
                    ),
                    bgcolor=Colors.with_opacity(0.2, Colors.BLACK),
                    expanded=initially_expanded,
                    content=Container(
                        content=text_widget,
                        padding=padding.only(left=16, right=16, bottom=12, top=4),
                        alignment=alignment.top_left,
                    ),
                ),
            ],
        )

    def _build_debug_section_empty(self, title: str, icon: str) -> Container:
        """Build a greyed-out, non-expandable placeholder for an empty debug section."""
        return Container(
            content=Row([
                Text(icon, size=14, opacity=0.35),
                Text(title, size=12, weight=FontWeight.BOLD, color=Colors.GREY_700),
                Text("— (empty)", size=10, color=Colors.GREY_800, italic=True),
            ], spacing=8, vertical_alignment=CrossAxisAlignment.CENTER),
            padding=padding.only(left=16, top=6, bottom=6),
        )

    def _build_rich_prompt_section(self, prompt_text: str,
                                     initially_expanded: bool = False) -> ft.ExpansionPanelList:
        """Build a color-coded collapsible section for the full formatted prompt.

        Parses [INST], [/INST], </s> tags and renders each segment in a
        distinct color with line breaks between turns for readability.
        """
        import re

        # Colour key
        TAG_COLOR = Colors.AMBER_400           # [INST], [/INST], </s>
        SYSTEM_COLOR = Colors.PINK_200         # First system block
        USER_COLOR = Colors.LIGHT_BLUE_400     # User messages
        LLM_COLOR = Colors.GREEN_200           # LLM replies
        LINEBREAK_COLOR = Colors.GREY_800      # Visual separator line

        # Split the prompt around tags, keeping the tags as separate tokens
        tokens = re.split(r'(\[INST\]|\[/INST\]|</s>)', prompt_text)

        spans: list[ft.TextSpan] = []
        # State: track what region we're in
        # "system_inst" = first [INST]...[/INST] pair (system prompt)
        # "user" = subsequent [INST]...[/INST] pairs
        # "assistant" = after [/INST] (non-first)
        inst_count = 0  # How many [INST] we've seen

        for tok in tokens:
            if not tok:
                continue
            if tok == "[INST]":
                inst_count += 1
                # Add a blank line before [INST] for readability (except the very first)
                if inst_count > 1:
                    spans.append(ft.TextSpan(
                        text="\n\n",
                        style=ft.TextStyle(size=11, font_family="Courier New", color=LINEBREAK_COLOR),
                    ))
                spans.append(ft.TextSpan(
                    text="[INST]",
                    style=ft.TextStyle(
                        size=11, font_family="Courier New", color=TAG_COLOR,
                        weight=FontWeight.BOLD,
                    ),
                ))
            elif tok == "[/INST]":
                # Add line break before [/INST] tag
                spans.append(ft.TextSpan(
                    text="\n",
                    style=ft.TextStyle(size=11, font_family="Courier New", color=LINEBREAK_COLOR),
                ))
                spans.append(ft.TextSpan(
                    text="[/INST]",
                    style=ft.TextStyle(
                        size=11, font_family="Courier New", color=TAG_COLOR,
                        weight=FontWeight.BOLD,
                    ),
                ))
                # Add line break after [/INST] tag
                spans.append(ft.TextSpan(
                    text="\n",
                    style=ft.TextStyle(size=11, font_family="Courier New", color=LINEBREAK_COLOR),
                ))
            elif tok == "</s>":
                spans.append(ft.TextSpan(
                    text="\n",
                    style=ft.TextStyle(size=11, font_family="Courier New", color=LINEBREAK_COLOR),
                ))
                spans.append(ft.TextSpan(
                    text="</s>",
                    style=ft.TextStyle(
                        size=11, font_family="Courier New", color=TAG_COLOR,
                        weight=FontWeight.BOLD,
                    ),
                ))
            else:
                # Determine colour based on context
                # After the first [INST] and before first [/INST] → system/user prompt
                # After [/INST] → assistant text
                # We look at preceding tags to figure out context
                # Simple heuristic: count preceding tags
                # If we just saw [INST] → user content (or system on first)
                # If we just saw [/INST] → assistant content
                last_tag = None
                for s in reversed(spans):
                    if s.text and s.text.strip() in ("[INST]", "[/INST]", "</s>"):
                        last_tag = s.text.strip()
                        break

                if last_tag == "[/INST]":
                    if inst_count == 1:
                        text_color = SYSTEM_COLOR  # "Understood." acknowledgment
                    else:
                        text_color = LLM_COLOR
                elif last_tag == "[INST]":
                    if inst_count == 1:
                        text_color = SYSTEM_COLOR
                    else:
                        text_color = USER_COLOR
                else:
                    text_color = Colors.GREY_300  # fallback

                spans.append(ft.TextSpan(
                    text=tok,
                    style=ft.TextStyle(size=11, font_family="Courier New", color=text_color),
                ))

        rich_text = ft.SelectionArea(
            content=ft.Text(
                spans=spans,
                selectable=True,
            ),
        )

        # Per-section copy button for the full prompt
        _prompt_text = prompt_text
        def _copy_prompt(e, _text=_prompt_text):
            self.page.set_clipboard(_text)
            self.page.snack_bar = SnackBar(content=Text("Copied: Full Formatted Prompt"))
            self.page.snack_bar.open = True
            self.page.update()

        prompt_copy_btn = IconButton(
            icon=Icons.COPY,
            icon_size=14,
            tooltip="Copy Full Formatted Prompt",
            on_click=_copy_prompt,
            style=ButtonStyle(padding=padding.all(0)),
        )

        return ft.ExpansionPanelList(
            elevation=0,
            expanded_header_padding=padding.all(0),
            controls=[
                ft.ExpansionPanel(
                    header=ft.ListTile(
                        leading=Text("📜", size=14),
                        title=Row([
                            Text("Full Formatted Prompt", size=12, weight=FontWeight.BOLD,
                                 color=Colors.AMBER_300),
                            prompt_copy_btn,
                        ], spacing=4, alignment=MainAxisAlignment.START),
                        dense=True,
                        content_padding=padding.only(left=8),
                    ),
                    bgcolor=Colors.with_opacity(0.2, Colors.BLACK),
                    expanded=initially_expanded,
                    content=Container(
                        content=Column([
                            rich_text,
                        ], spacing=0, horizontal_alignment=CrossAxisAlignment.START),
                        padding=padding.only(left=16, right=16, bottom=12, top=4),
                        alignment=alignment.top_left,
                    ),
                ),
            ],
        )

    def _populate_debug_console(self):
        """Populate the debug console with color-coded collapsible sections.
        
        ALL sections are always shown. Empty sections appear greyed-out and
        non-expandable so the user can see at a glance what was/wasn't present.
        """
        from core.chat_gen import _last_debug_info as dbg
        if not dbg:
            return
        
        # Hide placeholder text once we have data, show color key
        self.debug_no_data_text.visible = False
        self.debug_color_key.visible = True
        
        panels = []
        
        def _section(key_or_value, title, color, icon, expanded=False, value_override=None, custom_spans=None):
            """Append an expandable section if value exists, else a greyed-out placeholder."""
            value = value_override if value_override is not None else dbg.get(key_or_value, "")
            if value and str(value).strip():
                panels.append(self._build_debug_section(title, str(value), color, icon, expanded, custom_spans=custom_spans))
            else:
                panels.append(self._build_debug_section_empty(title, icon))
        
        # ──────────────────────────────────────────────────────
        # 1. PROMPT COMPONENTS — building blocks of the system prompt
        # ──────────────────────────────────────────────────────
        panels.append(
            Container(
                content=Text("▸ PROMPT COMPONENTS", size=10, weight=FontWeight.BOLD,
                             color=Colors.GREY_500),
                padding=padding.only(left=8, top=8, bottom=2),
            )
        )
        
        _section("_base_system", "Base System Prompt (Character)", Colors.CYAN_300, "👤", True)
        _section("_char_instructions", "Character Instructions (always sent)", Colors.ORANGE_200, "📋", False)
        
        # Persona — build display string if present
        if dbg.get("_persona_name"):
            persona_display = f"Name: {dbg['_persona_name']}\nBackground: {dbg.get('_persona_bg', '(none)')}"
        else:
            persona_display = ""
        _section(None, "Persona Context", Colors.PURPLE_200, "🎭", False, value_override=persona_display)
        
        # ──────────────────────────────────────────────────────
        # 2. SCENE INSTRUCTIONS — per-turn directions from the instruction box
        # ──────────────────────────────────────────────────────
        panels.append(
            Container(
                content=Text("▸ SCENE INSTRUCTIONS", size=10, weight=FontWeight.BOLD,
                             color=Colors.GREY_500),
                padding=padding.only(left=8, top=10, bottom=2),
            )
        )
        _section("_directions", "Instructions (this turn)", Colors.TEAL_300, "🎬", True)
        
        # ──────────────────────────────────────────────────────
        # 3. MEMORIES — semantic retrieval results
        # ──────────────────────────────────────────────────────
        panels.append(
            Container(
                content=Text("▸ MEMORIES", size=10, weight=FontWeight.BOLD,
                             color=Colors.GREY_500),
                padding=padding.only(left=8, top=10, bottom=2),
            )
        )
        
        mem_list = dbg.get("_memories") or []
        retrieval = dbg.get("_retrieval")
        if mem_list:
            if retrieval:
                lines = [
                    f"Query: \"{retrieval.query[:80]}{'...' if len(retrieval.query) > 80 else ''}\"",
                    f"Pool: {retrieval.total_available} total → {len(retrieval.selected)} selected, {len(retrieval.rejected)} rejected",
                    f"Token budget: ~{retrieval.budget_tokens_used}/{retrieval.budget_tokens_max}",
                    "",
                    "── SELECTED ──",
                ]
                for c in retrieval.selected:
                    pin_tag = " 📌" if c.pinned else ""
                    src_tag = f" [{c.source}]" if c.source else ""
                    lines.append(f"  {c.score:+.3f}  {c.text}{src_tag}{pin_tag}")
                
                if retrieval.rejected:
                    lines.append("")
                    lines.append("── REJECTED ──")
                    for c in retrieval.rejected[:10]:
                        src_tag = f" [{c.source}]" if c.source else ""
                        lines.append(f"  {c.score:+.3f}  {c.text}{src_tag}")
                    if len(retrieval.rejected) > 10:
                        lines.append(f"  ... and {len(retrieval.rejected) - 10} more")
                
                mem_display = "\n".join(lines)
                section_title = f"Memories ({len(retrieval.selected)}/{retrieval.total_available} selected)"
            else:
                mem_display = f"{len(mem_list)} memories injected (retrieval disabled):\n" + "\n".join(f"  • {m}" for m in mem_list)
                section_title = f"Memories ({len(mem_list)}) — all injected"
            
            mem_spans = self._build_memory_spans(mem_display)
            panels.append(self._build_debug_section(section_title, mem_display, Colors.YELLOW_300, "🧠", False, custom_spans=mem_spans))
        else:
            panels.append(self._build_debug_section_empty("Memories", "🧠"))
        
        # ──────────────────────────────────────────────────────
        # 4. CONVERSATION HISTORY — prior turns sent to the model
        # ──────────────────────────────────────────────────────
        history_turns = dbg.get("history_turns", [])
        if history_turns:
            history_lines = []
            turn_num = 0
            i = 0
            while i < len(history_turns):
                turn = history_turns[i]
                if turn.get("role") == "user":
                    turn_num += 1
                    history_lines.append(f"── Turn {turn_num} ──")
                    history_lines.append(f"  User: {turn.get('content', '')}")
                    if (i + 1) < len(history_turns) and history_turns[i + 1].get("role") == "assistant":
                        content = history_turns[i + 1].get("content", "")
                        if len(content) > 300:
                            content = content[:300] + "..."
                        history_lines.append(f"  Assistant: {content}")
                        i += 1
                    history_lines.append("")
                else:
                    role = turn.get("role", "unknown").capitalize()
                    content = turn.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    history_lines.append(f"  {role}: {content}")
                    history_lines.append("")
                i += 1
            
            hist_display = "\n".join(history_lines).rstrip()
            
            panels.append(
                Container(
                    content=Text(f"▸ CONVERSATION HISTORY ({turn_num} turns)", size=10, weight=FontWeight.BOLD,
                                 color=Colors.GREY_500),
                    padding=padding.only(left=8, top=10, bottom=2),
                )
            )
            hist_spans = self._build_history_spans(hist_display)
            panels.append(self._build_debug_section(
                f"Prior Chat Messages ({turn_num} turns)",
                hist_display,
                Colors.BLUE_200, "💬", False,
                custom_spans=hist_spans,
            ))
        else:
            panels.append(
                Container(
                    content=Text("▸ CONVERSATION HISTORY", size=10, weight=FontWeight.BOLD,
                                 color=Colors.GREY_500),
                    padding=padding.only(left=8, top=10, bottom=2),
                )
            )
            panels.append(self._build_debug_section_empty("Prior Chat Messages (0 turns)", "💬"))
        
        # ──────────────────────────────────────────────────────
        # 5. ASSEMBLED & SENT — the final system prompt + user message
        # ──────────────────────────────────────────────────────
        # Stats row
        history_len = dbg.get("history_length", 0)
        token_in = dbg.get("input_tokens", "?")
        token_out = dbg.get("output_tokens", "?")
        if retrieval:
            mem_label = f"🧠 Memories: {len(retrieval.selected)}/{retrieval.total_available} (~{retrieval.budget_tokens_used}t)"
            mem_color = Colors.YELLOW_300 if retrieval.selected else Colors.GREY_600
        else:
            mem_count = len(mem_list)
            mem_label = f"🧠 Memories: {mem_count}"
            mem_color = Colors.YELLOW_300 if mem_count else Colors.GREY_600
        stats_row = Container(
            content=Row([
                Text(f"📊 History: {history_len} turns", size=11, color=Colors.GREY_400),
                Text("│", size=11, color=Colors.GREY_700),
                Text(mem_label, size=11, color=mem_color),
                Text("│", size=11, color=Colors.GREY_700),
                Text(f"🔢 Tokens: {token_in} in → {token_out} out", size=11, color=Colors.GREY_400),
            ], spacing=10, wrap=True),
            padding=padding.symmetric(horizontal=12, vertical=6),
            bgcolor=Colors.with_opacity(0.1, Colors.WHITE),
            border_radius=4,
        )
        
        panels.append(
            Container(
                content=Text("▸ ASSEMBLED & SENT", size=10, weight=FontWeight.BOLD,
                             color=Colors.GREY_500),
                padding=padding.only(left=8, top=10, bottom=2),
            )
        )
        _section("system_prompt", "Assembled System Prompt", Colors.CYAN_100, "📋", False)
        _section("user_message", "User Message", Colors.LIGHT_BLUE_300, "💬", False)
        panels.append(stats_row)
        
        # ──────────────────────────────────────────────────────
        # 6. GENERATION PIPELINE — full prompt, response, cleanup
        # ──────────────────────────────────────────────────────
        full_prompt = dbg.get("full_prompt", "")
        
        # Build stop-phrases display (always present)
        all_stops = dbg.get("stop_phrases", [])
        if all_stops:
            stops_lines = []
            # Separate auto-generated name stops from config stops
            for s in all_stops:
                stops_lines.append(f"  {repr(s)}")
            stops_display = "\n".join(stops_lines)
        else:
            stops_display = "(none)"
        
        panels.append(
            Container(
                content=Text("▸ GENERATION PIPELINE", size=10, weight=FontWeight.BOLD,
                             color=Colors.GREY_500),
                padding=padding.only(left=8, top=10, bottom=2),
            )
        )
        # Rich color-coded full prompt
        if full_prompt:
            panels.append(self._build_rich_prompt_section(full_prompt))
        else:
            panels.append(self._build_debug_section_empty("Full Formatted Prompt", "📜"))
        
        _section(None, "Raw Response (pre-cleanup)", Colors.GREEN_300, "🟢", False,
                 value_override=dbg.get("raw_response", ""))
        stop_spans = self._build_stop_phrase_spans(stops_display) if stops_display != "(none)" else None
        _section(None, "Stop Phrases Applied", Colors.ORANGE_300, "🛑", False,
                 value_override=stops_display, custom_spans=stop_spans)
        _section(None, "Cleaned Response", Colors.LIGHT_GREEN_300, "✅", True,
                 value_override=dbg.get("cleaned_response", ""))
        
        if dbg.get("error"):
            panels.append(self._build_debug_section("Error", dbg["error"], Colors.RED_300, "❌", True))
        
        self.debug_panel_list.controls = panels
    def _build_message_spans(self, content: str) -> list:
        """
        Build a list of ft.TextSpan objects for rich-text message display.
        Action segments (*...*) are shown in italic muted color; dialog is normal.
        Returns (has_actions, spans).
        """
        from core.chat_gen import parse_message_with_actions
        parsed = parse_message_with_actions(content)
        if not parsed["has_actions"]:
            return False, [ft.TextSpan(text=content)]
        
        spans = []
        for i, seg in enumerate(parsed["segments"]):
            if seg["type"] == "action":
                spans.append(ft.TextSpan(
                    text=f"*{seg['text']}*",
                    style=ft.TextStyle(italic=True, color=Colors.GREY_400),
                ))
            else:
                spans.append(ft.TextSpan(text=seg["text"]))
            # Add a space between segments (not after the last one)
            if i < len(parsed["segments"]) - 1:
                spans.append(ft.TextSpan(text=" "))
        return True, spans

    def _add_message(self, role: str, content: str):
        """Add a message bubble to the chat."""
        is_user = role == "user"
        
        # Get display name from character/persona
        if is_user:
            display_name = self.selected_persona['name'] if self.selected_persona else "You"
        else:
            display_name = self.selected_character['name'] if self.selected_character else "AI"
        
        # Theme-aware colors: use primary for user, surface variant for AI
        if is_user:
            bubble_bg = Colors.with_opacity(0.3, Colors.PRIMARY)
            name_color = Colors.PRIMARY
        else:
            bubble_bg = Colors.with_opacity(0.15, Colors.ON_SURFACE)
            name_color = Colors.SECONDARY
        
        # Build message widget with action formatting
        has_actions, spans = self._build_message_spans(content)
        if has_actions:
            message_widget = ft.Text(spans=spans, selectable=True)
        else:
            message_widget = Text(content, selectable=True)
        
        bubble = Container(
            content=Column([
                Text(
                    display_name,
                    size=10,
                    weight=FontWeight.BOLD,
                    color=name_color,
                ),
                message_widget,
            ], spacing=3),
            bgcolor=bubble_bg,
            padding=padding.all(12),
            border_radius=border_radius.all(12),
            margin=ft.margin.only(
                left=50 if is_user else 0,
                right=0 if is_user else 50,
            ),
        )
        
        self.chat_messages.controls.append(bubble)
    
    def _send_message(self, e):
        """Send a message and get AI response."""
        if self.generating:
            return
        
        message = self.message_input.value.strip()
        if not message:
            return
        
        # Capture directions before clearing (directions are NOT stored in history)
        directions = self.directions_input.value.strip() if self.directions_input.value else ""
        
        # Check if a chat model is loaded
        if app_state.loaded_model_type != "chat":
            self._add_message("assistant", "⚠️ Please load a 💬 Chat model from the header dropdown first.")
            self.page.update()
            return
        
        # Commit any pending uncommitted response before sending new message
        self._commit_pending_response()
        
        # Create conversation in DB if this is the first message
        if self.current_conversation_id is None:
            model_name = app_state.loaded_model or "unknown"
            char_id = self.selected_character['id'] if self.selected_character else None
            persona_id = self.selected_persona['id'] if self.selected_persona else None
            title = message[:60] + ("..." if len(message) > 60 else "")
            self.current_conversation_id = self.db.create_conversation(
                model=model_name,
                ai_character_id=char_id,
                user_persona_id=persona_id,
                title=title,
            )
        
        # Save user message to DB
        self.db.add_message(self.current_conversation_id, "user", message)
        self.last_user_message = message
        
        # Lock dropdowns now that we have messages
        self._update_dropdown_lock()
        
        # Clear input and add user message
        self.message_input.value = ""
        self._add_message("user", message)
        self.generating = True
        self.typing_indicator.visible = True
        self.progress.visible = True
        self.send_btn.disabled = True
        self.suggest_prompt_btn.visible = False
        self.suggested_prompt_container.visible = False
        self.suggestion_responses = []
        self.suggestion_index = 0
        self.page.update()
        
        def do_generate():
            from core.prompt_builder import PromptBuilder
            
            # Build system prompt from character or manual input
            if self.selected_character:
                base_system = self.selected_character.get('system_prompt', '')
                temp = self.selected_character.get('temperature', 0.7)
            else:
                base_system = self.system_prompt.value or None
                temp = self.temp_slider.value
            
            # Prepare PromptBuilder with character overrides
            builder = PromptBuilder()
            builder.load_character_overrides(self.selected_character)
            
            # Gather persona info
            persona_name = None
            persona_bg = None
            if self.selected_persona and self.selected_persona.get('background'):
                persona_name = self.selected_persona['name']
                persona_bg = self.selected_persona['background']
            
            # Gather memories via semantic retrieval
            memory_list = None
            retrieval_result = None
            if self.current_conversation_id:
                all_memories = self.db.get_memories(self.current_conversation_id)
                if all_memories:
                    from core.semantic_index import retrieve_memories
                    from core.config import get_setting
                    retrieval_enabled = get_setting("memory_retrieval_enabled", True)
                    if retrieval_enabled:
                        retrieval_result = retrieve_memories(
                            query_text=message,
                            memories=all_memories,
                            top_k=get_setting("memory_top_k", 10),
                            min_score=get_setting("memory_min_score", 0.25),
                            max_token_budget=get_setting("memory_token_budget", 400),
                        )
                        memory_list = retrieval_result.selected_texts
                    else:
                        memory_list = [m['content'] for m in all_memories]
            
            # Assemble system prompt (persona, memories, directions all handled)
            system = builder.build_system_prompt(
                base_system_prompt=base_system,
                persona_name=persona_name,
                persona_background=persona_bg,
                directions=directions,
                memories=memory_list,
            )
            
            # Get stop phrases from builder
            char_name = self.selected_character.get('name') if self.selected_character else None
            p_name = self.selected_persona.get('name') if self.selected_persona else None
            extra_stops = builder.get_stop_phrases(character_name=char_name, persona_name=p_name)
            
            success, response = generate_chat_response(
                user_message=message,
                system_prompt=system,
                max_new_tokens=int(self.max_tokens_slider.value),
                temperature=temp,
                directions="",  # directions already injected via builder
                extra_stop_phrases=extra_stops,
            )
            
            # Capture prompt component details for debug console
            # (must be AFTER generate_chat_response, which clears _last_debug_info)
            import core.chat_gen as _dbg_mod
            _dbg_mod._last_debug_info["_persona_name"] = persona_name or ""
            _dbg_mod._last_debug_info["_persona_bg"] = persona_bg or ""
            _dbg_mod._last_debug_info["_memories"] = memory_list or []
            _dbg_mod._last_debug_info["_retrieval"] = retrieval_result
            _dbg_mod._last_debug_info["_directions"] = directions or ""
            _dbg_mod._last_debug_info["_base_system"] = base_system or ""
            _dbg_mod._last_debug_info["_char_instructions"] = builder._effective("character_instructions_template") or ""
            
            # generate_chat_response added user+assistant to _conversation_history
            # Pop the assistant - it stays uncommitted until user sends next message
            import core.chat_gen as _chat_mod
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "assistant":
                _chat_mod._conversation_history.pop()
            
            # Auto-extract memories from the exchange (runs in background, non-blocking)
            if success and self.current_conversation_id:
                try:
                    from core.chat_gen import extract_memories
                    existing_mems = [m['content'] for m in self.db.get_memories(self.current_conversation_id)]
                    # Pass the base system prompt (before memories were injected) so extraction
                    # knows what character/setting info is already provided and won't duplicate it
                    base_system = None
                    if self.selected_character:
                        base_system = self.selected_character.get('system_prompt', '')
                    elif self.system_prompt.value:
                        base_system = self.system_prompt.value
                    new_memories = extract_memories(message, response, existing_mems, system_prompt=base_system)
                    for mem in new_memories:
                        self.db.add_memory(self.current_conversation_id, mem, source="auto")
                    if new_memories:
                        logger.info(f"Auto-saved {len(new_memories)} memories")
                        self._update_memory_button()
                except Exception as ex:
                    logger.warning(f"Memory extraction skipped: {ex}")
            
            # Initialize regen carousel with this response (uncommitted)
            self.regen_responses = [response]
            self.regen_index = 0
            self.response_committed = False
            
            # Populate debug console with prompt info
            self._populate_debug_console()
            
            # Build carousel bubble (not a plain bubble)
            carousel_bubble = self._build_ai_carousel_bubble()
            self.chat_messages.controls.append(carousel_bubble)
            
            self.generating = False
            self.typing_indicator.visible = False
            self.progress.visible = False
            # Only re-enable send if there's text in the input
            has_text = bool(self.message_input.value and self.message_input.value.strip())
            self.send_btn.disabled = not has_text
            self.suggest_prompt_btn.visible = True
            
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _clear_conversation(self, e=None):
        """Clear the conversation UI and start fresh (no DB delete)."""
        clear_conversation()
        self.chat_messages.controls.clear()
        self.current_conversation_id = None
        self.last_user_message = None
        self.regen_responses = []
        self.regen_index = 0
        self.response_committed = True
        self.regen_text_ctrl = None
        self.regen_counter_text = None
        self.regen_prev_btn = None
        self.regen_next_btn = None
        self.suggest_prompt_btn.visible = False
        self.suggested_prompt_container.visible = False
        self.suggestion_responses = []
        self.suggestion_index = 0
        self.suggestion_text_ctrl = None
        self.suggestion_counter_text = None
        self.suggestion_prev_btn = None
        self.suggestion_next_btn = None
        self._update_memory_button()
        self._update_dropdown_lock()
        self.page.update()
    
    def _new_chat(self, e=None):
        """Start a new chat - commits pending response if any, then clears."""
        self._commit_pending_response()
        self._clear_conversation()
    
    # =========== MEMORY MANAGEMENT ===========
    
    def _show_memories_dialog(self, e=None):
        """Show a dialog listing all memories for the current conversation, with add/edit/delete."""
        if not self.current_conversation_id:
            self.page.snack_bar = SnackBar(
                content=Text("💡 Start a conversation first to see memories."),
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        memories = self.db.get_memories(self.current_conversation_id)
        
        memory_list = ListView(spacing=6, expand=True, padding=padding.all(4))
        new_memory_field = TextField(
            label="Add a new memory...",
            hint_text="e.g., User's name is Alex",
            multiline=True,
            min_lines=1,
            max_lines=3,
            expand=True,
        )
        
        def refresh_memory_list():
            """Rebuild the memory list from DB."""
            nonlocal memories
            memories = self.db.get_memories(self.current_conversation_id)
            memory_list.controls.clear()
            
            if not memories:
                memory_list.controls.append(
                    Container(
                        content=Text(
                            "No memories yet. Memories are auto-generated as you chat,\nor you can add them manually below.",
                            size=12, color=Colors.GREY_500, text_align=ft.TextAlign.CENTER,
                        ),
                        padding=padding.all(20),
                        alignment=alignment.center,
                    )
                )
            else:
                for mem in memories:
                    source_icon = "🤖" if mem['source'] == 'auto' else "✏️"
                    
                    memory_list.controls.append(
                        Container(
                            content=Row([
                                Text(source_icon, size=14),
                                Text(mem['content'], size=12, expand=True),
                                IconButton(
                                    icon=Icons.EDIT, icon_size=16, tooltip="Edit",
                                    on_click=lambda e, m=mem: _edit_memory(m),
                                ),
                                IconButton(
                                    icon=Icons.DELETE, icon_size=16, tooltip="Delete",
                                    icon_color=Colors.RED_400,
                                    on_click=lambda e, m=mem: _delete_memory(m['id']),
                                ),
                            ], spacing=6, vertical_alignment=CrossAxisAlignment.CENTER),
                            bgcolor=Colors.GREY_900,
                            padding=padding.symmetric(horizontal=10, vertical=6),
                            border_radius=border_radius.all(6),
                        )
                    )
            
            # Update memory button text with count
            count = len(memories)
            self.memory_btn.text = f"🧠 Memories ({count})" if count > 0 else "🧠 Memories"
            self.page.update()
        
        def _add_memory(e):
            text = new_memory_field.value.strip()
            if not text:
                return
            self.db.add_memory(self.current_conversation_id, text, source="manual")
            new_memory_field.value = ""
            refresh_memory_list()
        
        def _delete_memory(memory_id):
            self.db.delete_memory(memory_id)
            refresh_memory_list()
        
        def _edit_memory(mem):
            """Show inline edit for a memory."""
            edit_field = TextField(value=mem['content'], multiline=True, min_lines=1, max_lines=3, expand=True)
            
            def save_edit(e):
                new_text = edit_field.value.strip()
                if new_text:
                    self.db.update_memory(mem['id'], new_text)
                refresh_memory_list()
            
            def cancel_edit(e):
                refresh_memory_list()
            
            # Replace the memory item in the list with edit controls
            for i, ctrl in enumerate(memory_list.controls):
                if hasattr(ctrl, 'data') and ctrl.data == mem['id']:
                    memory_list.controls[i] = Container(
                        content=Row([
                            edit_field,
                            IconButton(icon=Icons.CHECK, icon_size=16, tooltip="Save", on_click=save_edit),
                            IconButton(icon=Icons.CLOSE, icon_size=16, tooltip="Cancel", on_click=cancel_edit),
                        ], spacing=4),
                        bgcolor=Colors.GREY_800,
                        padding=padding.symmetric(horizontal=10, vertical=6),
                        border_radius=border_radius.all(6),
                    )
                    self.page.update()
                    return
            
            # Fallback: rebuild with edit mode for the target memory
            memory_list.controls.clear()
            for m in memories:
                if m['id'] == mem['id']:
                    memory_list.controls.append(
                        Container(
                            content=Row([
                                edit_field,
                                IconButton(icon=Icons.CHECK, icon_size=16, tooltip="Save", on_click=save_edit),
                                IconButton(icon=Icons.CLOSE, icon_size=16, tooltip="Cancel", on_click=cancel_edit),
                            ], spacing=4),
                            bgcolor=Colors.GREY_800,
                            padding=padding.symmetric(horizontal=10, vertical=6),
                            border_radius=border_radius.all(6),
                        )
                    )
                else:
                    source_icon = "🤖" if m['source'] == 'auto' else "✏️"
                    memory_list.controls.append(
                        Container(
                            content=Row([
                                Text(source_icon, size=14),
                                Text(m['content'], size=12, expand=True),
                                IconButton(icon=Icons.EDIT, icon_size=16, tooltip="Edit", on_click=lambda e, mi=m: _edit_memory(mi)),
                                IconButton(icon=Icons.DELETE, icon_size=16, tooltip="Delete", icon_color=Colors.RED_400, on_click=lambda e, mid=m['id']: _delete_memory(mid)),
                            ], spacing=6, vertical_alignment=CrossAxisAlignment.CENTER),
                            bgcolor=Colors.GREY_900,
                            padding=padding.symmetric(horizontal=10, vertical=6),
                            border_radius=border_radius.all(6),
                        )
                    )
            self.page.update()
        
        def _delete_all_memories(e):
            self.db.delete_all_memories(self.current_conversation_id)
            refresh_memory_list()
        
        def close_dialog(e):
            dialog.open = False
            if dialog in self.page.overlay:
                self.page.overlay.remove(dialog)
            # Update the memory button count
            count = len(self.db.get_memories(self.current_conversation_id)) if self.current_conversation_id else 0
            self.memory_btn.text = f"🧠 Memories ({count})" if count > 0 else "🧠 Memories"
            self.page.update()
        
        # Build initial list
        refresh_memory_list()
        
        dialog = AlertDialog(
            title=Text("🧠 Conversation Memories"),
            content=Container(
                content=Column([
                    Text("Memories help the AI remember key facts across long conversations.\n🤖 = auto-generated  ✏️ = manually added", size=11, color=Colors.GREY_500),
                    Container(height=6),
                    Container(
                        content=memory_list,
                        height=300,
                        width=500,
                    ),
                    Container(height=8),
                    Row([
                        new_memory_field,
                        IconButton(icon=Icons.ADD_CIRCLE, icon_size=24, tooltip="Add memory", on_click=_add_memory),
                    ], spacing=6),
                    Row([
                        ElevatedButton("🗑️ Clear All", on_click=_delete_all_memories, color=Colors.RED_400,
                            height=28, style=ButtonStyle(padding=padding.symmetric(horizontal=8, vertical=2), text_style=ft.TextStyle(size=10))),
                    ], alignment=MainAxisAlignment.END),
                ], spacing=4),
                width=520,
            ),
            actions=[
                TextButton("Close", on_click=close_dialog),
            ],
        )
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _update_memory_button(self):
        """Update the memory button text with current count."""
        if self.current_conversation_id:
            count = len(self.db.get_memories(self.current_conversation_id))
            self.memory_btn.text = f"🧠 Memories ({count})" if count > 0 else "🧠 Memories"
        else:
            self.memory_btn.text = "🧠 Memories"
    
    def _delete_current_conversation(self, e=None):
        """Delete the current conversation from DB and clear UI."""
        if self.current_conversation_id:
            try:
                self.db.delete_conversation(self.current_conversation_id)
            except Exception as ex:
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Delete failed: {ex}"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
                self.page.update()
                return
        self._clear_conversation()
        self.page.snack_bar = SnackBar(content=Text("🗑️ Conversation deleted."))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _update_dropdown_lock(self):
        """Lock/unlock char and persona dropdowns based on active chat state."""
        should_lock = self.current_conversation_id is not None
        self.char_dropdown.disabled = should_lock
        self.persona_dropdown.disabled = should_lock
    
    # =========== REGEN CAROUSEL ===========
    
    def _build_ai_carousel_bubble(self):
        """Build the AI response bubble with regen carousel controls."""
        display_name = self.selected_character['name'] if self.selected_character else "AI"
        current_response = self.regen_responses[self.regen_index] if self.regen_responses else ""
        total = len(self.regen_responses)
        show_nav = total > 1
        
        # Build with action formatting
        has_actions, spans = self._build_message_spans(current_response)
        if has_actions:
            self.regen_text_ctrl = ft.Text(spans=spans, selectable=True)
        else:
            self.regen_text_ctrl = Text(current_response, selectable=True)
        self.regen_counter_text = Text(
            f"{self.regen_index + 1}/{total}",
            size=11, color=Colors.GREY_400,
            width=40, text_align=ft.TextAlign.CENTER,
            visible=show_nav,
        )
        self.regen_prev_btn = IconButton(
            icon=Icons.CHEVRON_LEFT, icon_size=16,
            tooltip="Previous response",
            on_click=self._regen_prev,
            disabled=self.regen_index == 0,
            visible=show_nav,
        )
        self.regen_next_btn = IconButton(
            icon=Icons.CHEVRON_RIGHT, icon_size=16,
            tooltip="Next response",
            on_click=self._regen_next,
            disabled=self.regen_index >= total - 1,
            visible=show_nav,
        )
        regen_inline_btn = IconButton(
            icon=Icons.REFRESH, icon_size=16,
            tooltip="Generate another response",
            on_click=self._regenerate_response,
            icon_color=Colors.BLUE_400,
        )
        
        carousel_row = Row([
            self.regen_prev_btn,
            self.regen_counter_text,
            self.regen_next_btn,
            Container(width=4),
            regen_inline_btn,
        ], spacing=0, alignment=MainAxisAlignment.START)
        
        bubble = Container(
            content=Column([
                Text(display_name, size=10, weight=FontWeight.BOLD, color=Colors.SECONDARY),
                self.regen_text_ctrl,
                carousel_row,
            ], spacing=3),
            bgcolor=Colors.with_opacity(0.15, Colors.ON_SURFACE),
            padding=padding.all(12),
            border_radius=border_radius.all(12),
            margin=ft.margin.only(left=0, right=50),
        )
        return bubble
    
    def _update_regen_display(self):
        """Update the carousel display with current regen index."""
        if not self.regen_responses or self.regen_text_ctrl is None:
            return
        current_response = self.regen_responses[self.regen_index]
        # Update text with action formatting
        has_actions, spans = self._build_message_spans(current_response)
        if has_actions:
            self.regen_text_ctrl.value = ""
            self.regen_text_ctrl.spans = spans
        else:
            self.regen_text_ctrl.value = current_response
            self.regen_text_ctrl.spans = []
        total = len(self.regen_responses)
        show_nav = total > 1
        self.regen_counter_text.value = f"{self.regen_index + 1}/{total}"
        self.regen_counter_text.visible = show_nav
        self.regen_prev_btn.visible = show_nav
        self.regen_next_btn.visible = show_nav
        self.regen_prev_btn.disabled = self.regen_index == 0
        self.regen_next_btn.disabled = self.regen_index >= total - 1
        try:
            self.page.update()
        except:
            pass
    
    def _regen_prev(self, e):
        """Show previous regen response."""
        if self.regen_index > 0:
            self.regen_index -= 1
            self._update_regen_display()
    
    def _regen_next(self, e):
        """Show next regen response."""
        if self.regen_index < len(self.regen_responses) - 1:
            self.regen_index += 1
            self._update_regen_display()
    
    def _commit_pending_response(self):
        """Commit the currently selected regen response to history and DB."""
        if self.response_committed or not self.regen_responses:
            return
        
        selected_response = self.regen_responses[self.regen_index]
        
        # Add to core conversation history
        import core.chat_gen as _chat_mod
        _chat_mod._conversation_history.append({"role": "assistant", "content": selected_response})
        
        # Save to DB
        if self.current_conversation_id:
            self.db.add_message(self.current_conversation_id, "assistant", selected_response)
        
        # Replace carousel bubble with plain bubble
        if self.chat_messages.controls:
            self.chat_messages.controls.pop()
            self._add_message("assistant", selected_response)
        
        # Reset regen state
        self.regen_responses = []
        self.regen_index = 0
        self.response_committed = True
        self.regen_text_ctrl = None
        self.regen_counter_text = None
        self.regen_prev_btn = None
        self.regen_next_btn = None
    
    def _regenerate_response(self, e):
        """Generate another AI response and add to carousel."""
        if self.generating or not self.last_user_message:
            return
        if app_state.loaded_model_type != "chat":
            return
        
        import core.chat_gen as _chat_mod
        
        # If response is currently committed, convert to carousel mode first
        if self.response_committed:
            # Get the committed assistant response from history
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "assistant":
                old_response = _chat_mod._conversation_history[-1]["content"]
                _chat_mod._conversation_history.pop()
            else:
                return  # No response to regenerate
            
            # Initialize carousel with the existing response
            self.regen_responses = [old_response]
            self.regen_index = 0
            self.response_committed = False
            
            # Replace the plain bubble with a carousel bubble
            if self.chat_messages.controls:
                self.chat_messages.controls.pop()
            carousel_bubble = self._build_ai_carousel_bubble()
            self.chat_messages.controls.append(carousel_bubble)
            self.page.update()
        
        # Now generate a new alternative response
        self.generating = True
        self.send_btn.disabled = True
        self.typing_indicator.visible = True
        self.page.update()
        
        def _do_regenerate():
            import core.chat_gen as _chat_mod
            from core.prompt_builder import PromptBuilder
            
            # Pop user from history (generate_chat_response will re-add it)
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "user":
                _chat_mod._conversation_history.pop()
            
            if self.selected_character:
                base_system = self.selected_character.get('system_prompt', '')
                temp = self.selected_character.get('temperature', 0.7)
            else:
                base_system = self.system_prompt.value or None
                temp = self.temp_slider.value
            
            # Prepare PromptBuilder with character overrides
            builder = PromptBuilder()
            builder.load_character_overrides(self.selected_character)
            
            # Gather persona info
            persona_name = None
            persona_bg = None
            if self.selected_persona and self.selected_persona.get('background'):
                persona_name = self.selected_persona['name']
                persona_bg = self.selected_persona['background']
            
            # Gather memories via semantic retrieval
            memory_list = None
            retrieval_result = None
            if self.current_conversation_id:
                all_memories = self.db.get_memories(self.current_conversation_id)
                if all_memories:
                    from core.semantic_index import retrieve_memories
                    from core.config import get_setting
                    retrieval_enabled = get_setting("memory_retrieval_enabled", True)
                    if retrieval_enabled:
                        retrieval_result = retrieve_memories(
                            query_text=self.last_user_message,
                            memories=all_memories,
                            top_k=get_setting("memory_top_k", 10),
                            min_score=get_setting("memory_min_score", 0.25),
                            max_token_budget=get_setting("memory_token_budget", 400),
                        )
                        memory_list = retrieval_result.selected_texts
                    else:
                        memory_list = [m['content'] for m in all_memories]
            
            # Assemble system prompt
            system = builder.build_system_prompt(
                base_system_prompt=base_system,
                persona_name=persona_name,
                persona_background=persona_bg,
                memories=memory_list,
            )
            
            # Get stop phrases from builder
            char_name = self.selected_character.get('name') if self.selected_character else None
            p_name = self.selected_persona.get('name') if self.selected_persona else None
            extra_stops = builder.get_stop_phrases(character_name=char_name, persona_name=p_name)
            
            success, response = generate_chat_response(
                user_message=self.last_user_message,
                system_prompt=system,
                max_new_tokens=int(self.max_tokens_slider.value),
                temperature=temp,
                extra_stop_phrases=extra_stops,
            )
            
            # Capture prompt component details for debug console
            # (must be AFTER generate_chat_response, which clears _last_debug_info)
            _chat_mod._last_debug_info["_persona_name"] = persona_name or ""
            _chat_mod._last_debug_info["_persona_bg"] = persona_bg or ""
            _chat_mod._last_debug_info["_memories"] = memory_list or []
            _chat_mod._last_debug_info["_retrieval"] = retrieval_result
            _chat_mod._last_debug_info["_directions"] = ""
            _chat_mod._last_debug_info["_base_system"] = base_system or ""
            _chat_mod._last_debug_info["_char_instructions"] = builder._effective("character_instructions_template") or ""
            
            # Pop assistant from history (keep user)
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "assistant":
                _chat_mod._conversation_history.pop()
            
            # Add to carousel and show latest
            self.regen_responses.append(response)
            self.regen_index = len(self.regen_responses) - 1
            
            # Populate debug console with prompt info
            self._populate_debug_console()
            
            # Update carousel display
            self._update_regen_display()
            
            self.generating = False
            has_text = bool(self.message_input.value and self.message_input.value.strip())
            self.send_btn.disabled = not has_text
            self.typing_indicator.visible = False
            
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=_do_regenerate, daemon=True).start()
    
    # =========== SUGGESTED RESPONSE ===========
    
    def _suggest_prompt(self, e):
        """Generate a suggested user response based on the full conversation context."""
        if self.generating:
            return
        if app_state.loaded_model_type != "chat":
            return
        
        self.generating = True
        self.suggest_prompt_btn.disabled = True
        self.page.update()
        
        def _do_suggest():
            import core.chat_gen as _chat_mod
            
            # Build conversation transcript for context
            history = _chat_mod._conversation_history.copy()
            
            # Include the pending (uncommitted) response if any
            if self.regen_responses and not self.response_committed:
                history.append({"role": "assistant", "content": self.regen_responses[self.regen_index]})
            
            # Build readable transcript (last 20 messages max)
            context_lines = []
            for msg in history[-20:]:
                role = "User" if msg["role"] == "user" else "AI"
                context_lines.append(f"{role}: {msg['content']}")
            context = "\n".join(context_lines)
            
            if context:
                suggestion_system = (
                    f"Here is the conversation so far:\n{context}\n\n"
                    "Based on this conversation, suggest a short, natural follow-up message "
                    "that the user could send next. Reply ONLY with the suggested message text, "
                    "nothing else. No quotes, no explanation, no preamble."
                )
            else:
                suggestion_system = (
                    "Suggest a short, interesting conversation starter message. "
                    "Reply ONLY with the suggested message text, nothing else."
                )
            
            success, suggestion = generate_chat_response(
                user_message="Suggest what the user should say next.",
                system_prompt=suggestion_system,
                max_new_tokens=100,
                temperature=0.9,
                use_history=False,
            )
            
            if success and suggestion.strip():
                suggested_text = suggestion.strip().strip('"').strip("'")
                # Add to suggestion carousel
                self.suggestion_responses.append(suggested_text)
                self.suggestion_index = len(self.suggestion_responses) - 1
                self._build_suggestion_display()
                self.suggested_prompt_container.visible = True
            
            self.generating = False
            self.suggest_prompt_btn.disabled = False
            
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=_do_suggest, daemon=True).start()
    
    def _build_suggestion_display(self):
        """Build/rebuild the suggestion carousel display."""
        if not self.suggestion_responses:
            return
        
        total = len(self.suggestion_responses)
        show_nav = total > 1
        current_text = self.suggestion_responses[self.suggestion_index]
        
        self.suggestion_text_ctrl = Text(current_text, size=13, selectable=True)
        self.suggestion_counter_text = Text(
            f"{self.suggestion_index + 1}/{total}",
            size=11, color=Colors.GREY_400,
            width=40, text_align=ft.TextAlign.CENTER,
            visible=show_nav,
        )
        self.suggestion_prev_btn = IconButton(
            icon=Icons.CHEVRON_LEFT, icon_size=14,
            tooltip="Previous suggestion",
            on_click=self._suggestion_prev,
            disabled=self.suggestion_index == 0,
            visible=show_nav,
        )
        self.suggestion_next_btn = IconButton(
            icon=Icons.CHEVRON_RIGHT, icon_size=14,
            tooltip="Next suggestion",
            on_click=self._suggestion_next,
            disabled=self.suggestion_index >= total - 1,
            visible=show_nav,
        )
        
        self.suggested_prompt_container.content = Column([
            Row([
                Text("💡 Suggested Response:", size=11, color=Colors.BLUE_300, italic=True),
                IconButton(
                    icon=Icons.REFRESH,
                    icon_size=14,
                    tooltip="Generate another suggestion",
                    on_click=self._suggest_prompt,
                ),
                IconButton(
                    icon=Icons.CLOSE,
                    icon_size=14,
                    tooltip="Dismiss",
                    on_click=lambda e: self._dismiss_suggestion(),
                ),
            ], spacing=4, alignment=MainAxisAlignment.START),
            Row([
                self.suggestion_text_ctrl,
                IconButton(
                    icon=Icons.ARROW_FORWARD,
                    icon_size=16,
                    tooltip="Use this response",
                    icon_color=Colors.BLUE_400,
                    on_click=lambda e: self._use_suggestion(self.suggestion_responses[self.suggestion_index]),
                ),
            ], spacing=4, expand=True, vertical_alignment=CrossAxisAlignment.START),
            Row([
                self.suggestion_prev_btn,
                self.suggestion_counter_text,
                self.suggestion_next_btn,
            ], spacing=0, alignment=MainAxisAlignment.START, visible=show_nav),
        ], spacing=2)
    
    def _update_suggestion_display(self):
        """Update suggestion carousel to show current index."""
        if not self.suggestion_responses or self.suggestion_text_ctrl is None:
            return
        total = len(self.suggestion_responses)
        show_nav = total > 1
        self.suggestion_text_ctrl.value = self.suggestion_responses[self.suggestion_index]
        self.suggestion_counter_text.value = f"{self.suggestion_index + 1}/{total}"
        self.suggestion_counter_text.visible = show_nav
        self.suggestion_prev_btn.visible = show_nav
        self.suggestion_next_btn.visible = show_nav
        self.suggestion_prev_btn.disabled = self.suggestion_index == 0
        self.suggestion_next_btn.disabled = self.suggestion_index >= total - 1
        try:
            self.page.update()
        except:
            pass
    
    def _suggestion_prev(self, e):
        """Show previous suggestion."""
        if self.suggestion_index > 0:
            self.suggestion_index -= 1
            self._update_suggestion_display()
    
    def _suggestion_next(self, e):
        """Show next suggestion."""
        if self.suggestion_index < len(self.suggestion_responses) - 1:
            self.suggestion_index += 1
            self._update_suggestion_display()
    
    def _use_suggestion(self, text: str):
        """Use a suggested response - put it in the message input."""
        self.message_input.value = text
        self.suggested_prompt_container.visible = False
        # Enable send since we just put text in
        self.send_btn.disabled = False
        self.page.update()
        self.message_input.focus()
    
    def _dismiss_suggestion(self):
        """Dismiss the suggested response display and reset state."""
        self.suggested_prompt_container.visible = False
        self.suggestion_responses = []
        self.suggestion_index = 0
        self.suggestion_text_ctrl = None
        self.suggestion_counter_text = None
        self.suggestion_prev_btn = None
        self.suggestion_next_btn = None
        self.page.update()
    
    # =========== HISTORY TAB ===========
    
    def _refresh_history_list(self, e=None):
        """Refresh the conversation history list."""
        self.history_list.controls.clear()
        search_term = self.history_search.value.strip().lower() if self.history_search.value else ""
        
        conversations = self.db.get_recent_conversations(limit=100)
        
        if not conversations:
            self.history_list.controls.append(
                Container(
                    content=Text("No conversations yet. Start chatting!", 
                                 size=14, color=Colors.GREY_500, italic=True),
                    padding=padding.all(20),
                    alignment=alignment.center,
                )
            )
            try:
                self.page.update()
            except:
                pass
            return
        
        for conv in conversations:
            title = conv.get('title', 'Untitled')
            model = conv.get('model', 'unknown')
            updated = conv.get('updated_at', '')
            conv_id = conv['id']
            
            # Look up character and persona names
            char_name = None
            persona_name = None
            if conv.get('ai_character_id'):
                char = self.db.get_ai_character(conv['ai_character_id'])
                char_name = f"{char['avatar']} {char['name']}" if char else "❓ Deleted"
            if conv.get('user_persona_id'):
                persona = self.db.get_user_persona(conv['user_persona_id'])
                persona_name = f"{persona['avatar']} {persona['name']}" if persona else "❓ Deleted"
            
            # Filter by search
            if search_term and search_term not in title.lower() and search_term not in model.lower():
                if not (char_name and search_term in char_name.lower()):
                    if not (persona_name and search_term in persona_name.lower()):
                        continue
            
            # Highlight if this is the active conversation
            is_active = conv_id == self.current_conversation_id
            bg_color = Colors.with_opacity(0.3, Colors.BLUE_700) if is_active else Colors.GREY_900
            
            # Format date  
            date_display = updated[:16] if updated else ""
            
            # Build info row items
            info_items = [
                Text(f"🤖 {model}", size=10, color=Colors.GREY_500),
            ]
            if char_name:
                info_items.append(Text(f"🎭 {char_name}", size=10, color=Colors.GREY_500))
            if persona_name:
                info_items.append(Text(f"👤 {persona_name}", size=10, color=Colors.GREY_500))
            info_items.append(Text(f"📅 {date_display}", size=10, color=Colors.GREY_500))
            
            entry = Container(
                content=Row([
                    Column([
                        Text(title, weight=FontWeight.BOLD, size=13, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        Row(info_items, spacing=10, wrap=True),
                    ], spacing=2, expand=True),
                    Row([
                        IconButton(
                            icon=Icons.OPEN_IN_NEW,
                            icon_size=18,
                            tooltip="Load conversation",
                            on_click=lambda e, cid=conv_id: self._load_conversation(cid),
                        ),
                        IconButton(
                            icon=Icons.DOWNLOAD,
                            icon_size=18,
                            tooltip="Export conversation",
                            on_click=lambda e, cid=conv_id: self._export_conversation(cid),
                        ),
                        IconButton(
                            icon=Icons.DELETE_OUTLINE,
                            icon_size=18,
                            tooltip="Delete conversation",
                            on_click=lambda e, cid=conv_id, t=title: self._confirm_delete_conversation(cid, t),
                            icon_color=Colors.RED_400,
                        ),
                    ], spacing=0),
                ], spacing=8),
                bgcolor=bg_color,
                padding=padding.all(10),
                border_radius=border_radius.all(8),
                border=border.all(1, Colors.BLUE_400) if is_active else None,
                on_click=lambda e, cid=conv_id: self._load_conversation(cid),
                ink=True,
            )
            self.history_list.controls.append(entry)
        
        try:
            self.page.update()
        except:
            pass
    
    def _load_conversation(self, conversation_id: int):
        """Load a prior conversation into the chat view, restoring character, persona, and model."""
        # Clear current chat UI and core history
        clear_conversation()
        self.chat_messages.controls.clear()
        self.current_conversation_id = conversation_id
        self.last_user_message = None
        self.regen_responses = []
        self.regen_index = 0
        self.response_committed = True
        self.regen_text_ctrl = None
        self.regen_counter_text = None
        self.regen_prev_btn = None
        self.regen_next_btn = None
        self.suggestion_responses = []
        self.suggestion_index = 0
        self.suggestion_text_ctrl = None
        self.suggested_prompt_container.visible = False
        
        # Get conversation metadata
        conv = self.db.get_conversation(conversation_id)
        warnings = []
        
        # Restore character
        if conv and conv.get('ai_character_id'):
            char = self.db.get_ai_character(conv['ai_character_id'])
            if char:
                self.selected_character = char
                self.char_dropdown.value = str(char['id'])
                self.system_prompt.visible = False
                self.temp_slider.value = char.get('temperature', 0.7)
            else:
                self.selected_character = None
                self.char_dropdown.value = "none"
                self.system_prompt.visible = True
                warnings.append("⚠️ The AI character used in this conversation no longer exists. Defaulting to None.")
        else:
            self.selected_character = None
            self.char_dropdown.value = "none"
            self.system_prompt.visible = True
        
        # Restore persona
        if conv and conv.get('user_persona_id'):
            persona = self.db.get_user_persona(conv['user_persona_id'])
            if persona:
                self.selected_persona = persona
                self.persona_dropdown.value = str(persona['id'])
            else:
                self.selected_persona = None
                self.persona_dropdown.value = "none"
                warnings.append("⚠️ The user persona used in this conversation no longer exists. Defaulting to None.")
        else:
            self.selected_persona = None
            self.persona_dropdown.value = "none"
        
        # Check if model needs to be loaded
        conv_model = conv.get('model', '') if conv else ''
        if conv_model and conv_model != 'unknown':
            current_model = app_state.loaded_model
            current_type = app_state.loaded_model_type
            
            if current_model != conv_model or current_type != "chat":
                # Model differs - show warning popup that we need to switch
                self._show_model_switch_dialog(conv_model, current_model)
            # If same model already loaded, nothing to do
        
        # Get messages from DB
        message_pairs = self.db.get_conversation_messages(conversation_id)
        
        if not message_pairs:
            self._add_message("assistant", "📂 This conversation is empty.")
            self._update_dropdown_lock()
            self.page.update()
            return
        
        # Rebuild core conversation history and UI
        import core.chat_gen as _chat_mod
        _chat_mod._conversation_history = []
        
        for user_msg, assistant_msg in message_pairs:
            _chat_mod._conversation_history.append({"role": "user", "content": user_msg})
            _chat_mod._conversation_history.append({"role": "assistant", "content": assistant_msg})
            self._add_message("user", user_msg)
            self._add_message("assistant", assistant_msg)
            self.last_user_message = user_msg
        
        # Show suggest button since we have history
        self.suggest_prompt_btn.visible = True
        
        # Update memory button count for loaded conversation
        self._update_memory_button()
        
        # Lock dropdowns since we have messages
        self._update_dropdown_lock()
        
        # Refresh history list to show active highlight
        self._refresh_history_list()
        
        # Show any warnings as popup
        if warnings:
            warning_dialog = AlertDialog(
                title=Text("⚠️ Loading Warnings"),
                content=Column([Text(w) for w in warnings], spacing=8, tight=True),
                actions=[TextButton("OK", on_click=lambda e: self._close_dialog(warning_dialog))],
            )
            self.page.overlay.append(warning_dialog)
            warning_dialog.open = True
        
        self.page.snack_bar = SnackBar(content=Text("📂 Conversation loaded!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _close_dialog(self, dialog):
        """Helper to close an AlertDialog."""
        dialog.open = False
        try:
            self.page.update()
        except:
            pass
    
    def _show_model_switch_dialog(self, needed_model: str, current_model: str):
        """Show a popup warning the user they need to load a different model."""
        if current_model:
            msg = f"This conversation used model '{needed_model}', but '{current_model}' is currently loaded.\n\nPlease load '{needed_model}' from the header dropdown to continue chatting."
        else:
            msg = f"This conversation used model '{needed_model}', but no model is currently loaded.\n\nPlease load '{needed_model}' from the header dropdown to continue chatting."
        
        dialog = AlertDialog(
            title=Text("🤖 Model Mismatch"),
            content=Text(msg),
            actions=[TextButton("OK", on_click=lambda e: self._close_dialog(dialog))],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
    
    def _export_conversation(self, conversation_id: int):
        """Export a conversation to JSON file using a file picker dialog."""
        import json
        
        try:
            data = self.db.export_conversation(conversation_id)
            self._pending_export_data = data
            
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in data.get('title', 'export'))
            filename = f"{safe_title}_{conversation_id}.json"
            
            # Default to user's Downloads folder
            downloads_dir = str(Path.home() / "Downloads")
            
            self.export_file_picker.save_file(
                dialog_title="Export Conversation",
                file_name=filename,
                initial_directory=downloads_dir,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["json"],
            )
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Export failed: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _on_export_file_picked(self, e: ft.FilePickerResultEvent):
        """Handle file picker result for export."""
        import json
        
        if e.path and self._pending_export_data:
            try:
                filepath = e.path
                # Ensure .json extension
                if not filepath.endswith('.json'):
                    filepath += '.json'
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self._pending_export_data, f, indent=2, ensure_ascii=False)
                
                self.page.snack_bar = SnackBar(
                    content=Text(f"✅ Exported to {filepath}"),
                )
                self.page.snack_bar.open = True
            except Exception as ex:
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Export failed: {ex}"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
            finally:
                self._pending_export_data = None
                self.page.update()
        else:
            self._pending_export_data = None
    
    def _confirm_delete_conversation(self, conversation_id: int, title: str):
        """Show confirmation dialog before deleting a conversation."""
        def close_dialog(e):
            dialog.open = False
            self.page.update()
        
        def do_delete(e):
            dialog.open = False
            self._delete_conversation(conversation_id)
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Conversation?"),
            content=Text(f"Delete \"{title}\"?\n\nThis cannot be undone."),
            actions=[
                TextButton("Cancel", on_click=close_dialog),
                TextButton("Delete", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _delete_conversation(self, conversation_id: int):
        """Delete a conversation from the database."""
        try:
            # If we're deleting the active conversation, clear the chat
            if conversation_id == self.current_conversation_id:
                clear_conversation()
                self.chat_messages.controls.clear()
                self.current_conversation_id = None
                self.last_user_message = None
                self.regen_responses = []
                self.regen_index = 0
                self.response_committed = True
                self.regen_text_ctrl = None
                self.suggest_prompt_btn.visible = False
                self.suggested_prompt_container.visible = False
                self.suggestion_responses = []
                self.suggestion_index = 0
                self.suggestion_text_ctrl = None
                self._update_dropdown_lock()
            
            self.db.delete_conversation(conversation_id)
            self._refresh_history_list()
            
            self.page.snack_bar = SnackBar(content=Text("🗑️ Conversation deleted."))
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Delete failed: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _confirm_delete_all_conversations(self, e):
        """Show confirmation dialog before deleting all conversations."""
        conversations = self.db.get_recent_conversations(limit=10000)
        count = len(conversations)
        if count == 0:
            self.page.snack_bar = SnackBar(content=Text("No conversations to delete."))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        def close_dialog(e):
            dialog.open = False
            self.page.update()
        
        def do_delete(e):
            dialog.open = False
            self._delete_all_conversations()
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete All Conversations?"),
            content=Text(f"This will permanently delete all {count} conversation(s).\n\nThis cannot be undone."),
            actions=[
                TextButton("Cancel", on_click=close_dialog),
                TextButton("Delete All", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _delete_all_conversations(self):
        """Delete all conversations from the database."""
        try:
            # If active conversation exists, clear it
            if self.current_conversation_id:
                clear_conversation()
                self.chat_messages.controls.clear()
                self.current_conversation_id = None
                self.last_user_message = None
                self.regen_responses = []
                self.regen_index = 0
                self.response_committed = True
                self.regen_text_ctrl = None
                self.suggest_prompt_btn.visible = False
                self.suggested_prompt_container.visible = False
                self.suggestion_responses = []
                self.suggestion_index = 0
                self.suggestion_text_ctrl = None
                self._update_dropdown_lock()
            
            self.db.delete_all_conversations()
            self._refresh_history_list()
            
            self.page.snack_bar = SnackBar(content=Text("🗑️ All conversations deleted."))
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Delete all failed: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    # =========== CHARACTER MANAGEMENT ===========
    
    def _build_char_list(self):
        """Build the character browse list."""
        self.char_list.controls.clear()
        
        # Add "Create New" button
        self.char_list.controls.append(
            Container(
                content=Row([
                    ft.Icon(Icons.ADD_CIRCLE_OUTLINE, size=24, color=Colors.BLUE_400),
                    Text("➕ Create New Character", size=14, weight=FontWeight.BOLD),
                ], spacing=10),
                bgcolor=Colors.BLUE_900,
                padding=padding.all(12),
                border_radius=border_radius.all(8),
                on_click=lambda e: self._edit_character(None),
                ink=True,
            )
        )
        
        # Add existing characters
        for char in self.db.get_all_ai_characters():
            self.char_list.controls.append(
                Container(
                    content=Row([
                        Text(char['avatar'], size=24),
                        Column([
                            Text(char['name'], weight=FontWeight.BOLD),
                            Text(char.get('description', '')[:50] or "No description", size=11, color=Colors.GREY_500),
                        ], spacing=2, expand=True),
                        Row([
                            IconButton(
                                icon=Icons.COPY, icon_size=18,
                                tooltip="Clone character",
                                on_click=lambda e, c=char: self._clone_character(c),
                                icon_color=Colors.BLUE_400,
                            ),
                            IconButton(
                                icon=Icons.DOWNLOAD, icon_size=18,
                                tooltip="Export as JSON",
                                on_click=lambda e, c=char: self._export_character(c),
                                icon_color=Colors.GREEN_400,
                            ),
                            IconButton(
                                icon=Icons.EDIT, icon_size=18,
                                tooltip="Edit character",
                                on_click=lambda e, c=char: self._edit_character(c),
                                icon_color=Colors.GREY_400,
                            ),
                        ], spacing=0),
                    ], spacing=10),
                    bgcolor=Colors.GREY_900,
                    padding=padding.all(12),
                    border_radius=border_radius.all(8),
                    on_click=lambda e, c=char: self._edit_character(c),
                    ink=True,
                )
            )
    
    def _show_char_browse(self, e=None):
        """Show character browse view."""
        self._build_char_list()
        self.char_browse_view.visible = True
        self.char_edit_view.visible = False
        self.page.update()
    
    def _edit_character(self, char):
        """Show character edit view."""
        self.editing_character_id = char['id'] if char else None
        
        # Populate form
        self.char_name_input.value = char['name'] if char else ""
        self.char_desc_input.value = char.get('description', '') if char else ""
        self.char_system_input.value = char.get('system_prompt', '') if char else ""
        self.char_avatar_input.value = char.get('avatar', '🤖') if char else "🤖"
        self.char_temp_slider.value = char.get('temperature', 0.7) if char else 0.7
        self.char_topp_slider.value = char.get('top_p', 0.95) if char else 0.95
        self.char_topk_slider.value = char.get('top_k', 50) if char else 50
        
        # Populate prompt override fields
        overrides = (char.get('prompt_overrides') or {}) if char else {}
        self.char_override_mode.value = overrides.get('override_mode', 'replace')
        self.char_override_char_instructions.value = overrides.get('character_instructions_template', '')
        self.char_override_memory.value = overrides.get('memory_template', '')
        
        self.char_delete_btn.visible = char is not None
        self.char_status.value = ""
        
        self.char_browse_view.visible = False
        self.char_edit_view.visible = True
        self.page.update()
    
    def _save_character(self, e):
        """Save character to database."""
        name = self.char_name_input.value.strip()
        if not name:
            self.char_status.value = "❌ Name is required"
            self.char_status.color = Colors.RED_400
            self.page.update()
            return
        
        data = {
            'name': name,
            'description': self.char_desc_input.value.strip(),
            'system_prompt': self.char_system_input.value.strip(),
            'avatar': self.char_avatar_input.value.strip() or "🤖",
            'temperature': self.char_temp_slider.value,
            'top_p': self.char_topp_slider.value,
            'top_k': int(self.char_topk_slider.value),
        }
        
        # Build prompt overrides dict (only include non-empty values)
        overrides = {}
        if self.char_override_char_instructions.value and self.char_override_char_instructions.value.strip():
            overrides['character_instructions_template'] = self.char_override_char_instructions.value.strip()
        if self.char_override_memory.value and self.char_override_memory.value.strip():
            overrides['memory_template'] = self.char_override_memory.value.strip()
        if overrides:
            overrides['override_mode'] = self.char_override_mode.value or 'replace'
        data['prompt_overrides'] = overrides if overrides else None
        
        try:
            if self.editing_character_id:
                self.db.update_ai_character(self.editing_character_id, **data)
                self.char_status.value = "✅ Character updated!"
            else:
                self.db.create_ai_character(**data)
                self.char_status.value = "✅ Character created!"
            
            self.char_status.color = Colors.GREEN_400
            self._refresh_dropdowns()
            self.page.update()
            
            # Go back to browse after a short delay
            import time
            time.sleep(0.5)
            self._show_char_browse()
        except Exception as ex:
            self.char_status.value = f"❌ Error: {ex}"
            self.char_status.color = Colors.RED_400
            self.page.update()
    
    def _delete_character(self, e):
        """Delete the current character."""
        if self.editing_character_id:
            try:
                self.db.delete_ai_character(self.editing_character_id)
                self._refresh_dropdowns()
                self._show_char_browse()
            except Exception as ex:
                self.char_status.value = f"❌ Error: {ex}"
                self.char_status.color = Colors.RED_400
                self.page.update()
    
    # =========== PERSONA MANAGEMENT ===========
    
    def _build_persona_list(self):
        """Build the persona browse list."""
        self.persona_list.controls.clear()
        
        # Add "Create New" button
        self.persona_list.controls.append(
            Container(
                content=Row([
                    ft.Icon(Icons.ADD_CIRCLE_OUTLINE, size=24, color=Colors.PURPLE_400),
                    Text("➕ Create New Persona", size=14, weight=FontWeight.BOLD),
                ], spacing=10),
                bgcolor=Colors.PURPLE_900,
                padding=padding.all(12),
                border_radius=border_radius.all(8),
                on_click=lambda e: self._edit_persona(None),
                ink=True,
            )
        )
        
        # Add existing personas
        for persona in self.db.get_all_user_personas():
            self.persona_list.controls.append(
                Container(
                    content=Row([
                        Text(persona['avatar'], size=24),
                        Column([
                            Text(persona['name'], weight=FontWeight.BOLD),
                            Text(persona.get('description', '')[:50] or "No description", size=11, color=Colors.GREY_500),
                        ], spacing=2, expand=True),
                        Row([
                            IconButton(
                                icon=Icons.COPY, icon_size=18,
                                tooltip="Clone persona",
                                on_click=lambda e, p=persona: self._clone_persona(p),
                                icon_color=Colors.PURPLE_400,
                            ),
                            IconButton(
                                icon=Icons.DOWNLOAD, icon_size=18,
                                tooltip="Export as JSON",
                                on_click=lambda e, p=persona: self._export_persona(p),
                                icon_color=Colors.GREEN_400,
                            ),
                            IconButton(
                                icon=Icons.EDIT, icon_size=18,
                                tooltip="Edit persona",
                                on_click=lambda e, p=persona: self._edit_persona(p),
                                icon_color=Colors.GREY_400,
                            ),
                        ], spacing=0),
                    ], spacing=10),
                    bgcolor=Colors.GREY_900,
                    padding=padding.all(12),
                    border_radius=border_radius.all(8),
                    on_click=lambda e, p=persona: self._edit_persona(p),
                    ink=True,
                )
            )
    
    def _show_persona_browse(self, e=None):
        """Show persona browse view."""
        self._build_persona_list()
        self.persona_browse_view.visible = True
        self.persona_edit_view.visible = False
        self.page.update()
    
    def _edit_persona(self, persona):
        """Show persona edit view."""
        self.editing_persona_id = persona['id'] if persona else None
        
        # Populate form
        self.persona_name_input.value = persona['name'] if persona else ""
        self.persona_desc_input.value = persona.get('description', '') if persona else ""
        self.persona_bg_input.value = persona.get('background', '') if persona else ""
        self.persona_avatar_input.value = persona.get('avatar', '👤') if persona else "👤"
        
        self.persona_delete_btn.visible = persona is not None
        self.persona_status.value = ""
        
        self.persona_browse_view.visible = False
        self.persona_edit_view.visible = True
        self.page.update()
    
    def _save_persona(self, e):
        """Save persona to database."""
        name = self.persona_name_input.value.strip()
        if not name:
            self.persona_status.value = "❌ Name is required"
            self.persona_status.color = Colors.RED_400
            self.page.update()
            return
        
        data = {
            'name': name,
            'description': self.persona_desc_input.value.strip(),
            'background': self.persona_bg_input.value.strip(),
            'avatar': self.persona_avatar_input.value.strip() or "👤",
        }
        
        try:
            if self.editing_persona_id:
                self.db.update_user_persona(self.editing_persona_id, **data)
                self.persona_status.value = "✅ Persona updated!"
            else:
                self.db.create_user_persona(**data)
                self.persona_status.value = "✅ Persona created!"
            
            self.persona_status.color = Colors.GREEN_400
            self._refresh_dropdowns()
            self.page.update()
            
            # Go back to browse after a short delay
            import time
            time.sleep(0.5)
            self._show_persona_browse()
        except Exception as ex:
            self.persona_status.value = f"❌ Error: {ex}"
            self.persona_status.color = Colors.RED_400
            self.page.update()
    
    def _delete_persona(self, e):
        """Delete the current persona."""
        if self.editing_persona_id:
            try:
                self.db.delete_user_persona(self.editing_persona_id)
                self._refresh_dropdowns()
                self._show_persona_browse()
            except Exception as ex:
                self.persona_status.value = f"❌ Error: {ex}"
                self.persona_status.color = Colors.RED_400
                self.page.update()
    
    # =========== CLONE & EXPORT ===========
    
    def _clone_character(self, char):
        """Clone an AI character with '- Clone' suffix (auto-increments if name taken)."""
        try:
            base_name = f"{char['name']} - Clone"
            clone_name = base_name
            existing = [c['name'] for c in self.db.get_all_ai_characters()]
            counter = 2
            while clone_name in existing:
                clone_name = f"{base_name} {counter}"
                counter += 1
            new_id = self.db.create_ai_character(
                name=clone_name,
                system_prompt=char.get('system_prompt', ''),
                temperature=char.get('temperature', 0.7),
                top_p=char.get('top_p', 0.95),
                top_k=char.get('top_k', 50),
                description=char.get('description', ''),
                avatar=char.get('avatar', '🤖'),
            )
            self._refresh_dropdowns()
            self._build_char_list()
            self.page.snack_bar = SnackBar(content=Text(f"✅ Cloned '{char['name']}' as '{clone_name}'"))
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Clone failed: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _clone_persona(self, persona):
        """Clone a user persona with '- Clone' suffix (auto-increments if name taken)."""
        try:
            base_name = f"{persona['name']} - Clone"
            clone_name = base_name
            existing = [p['name'] for p in self.db.get_all_user_personas()]
            counter = 2
            while clone_name in existing:
                clone_name = f"{base_name} {counter}"
                counter += 1
            new_id = self.db.create_user_persona(
                name=clone_name,
                description=persona.get('description', ''),
                background=persona.get('background', ''),
                avatar=persona.get('avatar', '👤'),
            )
            self._refresh_dropdowns()
            self._build_persona_list()
            self.page.snack_bar = SnackBar(content=Text(f"✅ Cloned '{persona['name']}' as '{clone_name}'"))
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Clone failed: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _export_character(self, char):
        """Export an AI character as JSON using file picker."""
        import json
        
        export_data = {
            "type": "ai_character",
            "name": char['name'],
            "description": char.get('description', ''),
            "system_prompt": char.get('system_prompt', ''),
            "temperature": char.get('temperature', 0.7),
            "top_p": char.get('top_p', 0.95),
            "top_k": char.get('top_k', 50),
            "avatar": char.get('avatar', '🤖'),
            "prompt_overrides": char.get('prompt_overrides'),
        }
        
        self._pending_export_data = export_data
        safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in char['name'])
        filename = f"character_{safe_name}.json"
        downloads_dir = str(Path.home() / "Downloads")
        
        self.export_file_picker.save_file(
            dialog_title="Export Character",
            file_name=filename,
            initial_directory=downloads_dir,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
    
    def _export_persona(self, persona):
        """Export a user persona as JSON using file picker."""
        import json
        
        export_data = {
            "type": "user_persona",
            "name": persona['name'],
            "description": persona.get('description', ''),
            "background": persona.get('background', ''),
            "avatar": persona.get('avatar', '👤'),
        }
        
        self._pending_export_data = export_data
        safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in persona['name'])
        filename = f"persona_{safe_name}.json"
        downloads_dir = str(Path.home() / "Downloads")
        
        self.export_file_picker.save_file(
            dialog_title="Export Persona",
            file_name=filename,
            initial_directory=downloads_dir,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
    
    # =========== PROMPT TEMPLATES MANAGEMENT ===========
    
    def _load_prompt_templates_ui(self):
        """Populate the prompt templates tab fields from config."""
        from core.prompt_builder import load_prompt_templates, _BUILTIN_DEFAULTS
        templates = load_prompt_templates()
        self.pt_char_instructions_input.value = templates.get("character_instructions_template", "")
        self.pt_persona_input.value = templates.get("persona_context_template", "")
        self.pt_directions_input.value = templates.get("directions_template", "")
        self.pt_memory_input.value = templates.get("memory_template", "")
        
        # Stop phrases: convert list to newline-separated text
        stop_list = templates.get("stop_phrases", _BUILTIN_DEFAULTS["stop_phrases"])
        self.pt_stop_phrases_input.value = "\n".join(stop_list) if isinstance(stop_list, list) else str(stop_list)
        
        self.pt_char_name_stop.value = templates.get("use_character_name_as_stop", True)
        self.pt_persona_name_stop.value = templates.get("use_persona_name_as_stop", True)
    
    def _save_prompt_templates(self, e):
        """Save prompt templates to config YAML."""
        from core.prompt_builder import save_prompt_templates
        
        # Parse stop phrases from newline-separated text
        raw_stops = self.pt_stop_phrases_input.value or ""
        stop_list = [line for line in raw_stops.split("\n") if line.strip()]
        
        templates = {
            "character_instructions_template": self.pt_char_instructions_input.value or "",
            "persona_context_template": self.pt_persona_input.value or "",
            "directions_template": self.pt_directions_input.value or "",
            "memory_template": self.pt_memory_input.value or "",
            "stop_phrases": stop_list,
            "use_character_name_as_stop": self.pt_char_name_stop.value,
            "use_persona_name_as_stop": self.pt_persona_name_stop.value,
        }
        
        if save_prompt_templates(templates):
            self.pt_status.value = "✅ Templates saved!"
            self.pt_status.color = Colors.GREEN_400
        else:
            self.pt_status.value = "❌ Failed to save"
            self.pt_status.color = Colors.RED_400
        self.page.update()
    
    def _reset_prompt_templates(self, e):
        """Reset prompt templates to built-in defaults."""
        from core.prompt_builder import _BUILTIN_DEFAULTS, save_prompt_templates
        
        if save_prompt_templates(dict(_BUILTIN_DEFAULTS)):
            self._load_prompt_templates_ui()
            self.pt_status.value = "✅ Reset to defaults!"
            self.pt_status.color = Colors.GREEN_400
        else:
            self.pt_status.value = "❌ Reset failed"
            self.pt_status.color = Colors.RED_400
        self.page.update()
    
    def _on_chat_tab_change(self, e):
        """Handle sub-tab changes - refresh data when switching tabs."""
        tab_index = e.control.selected_index if hasattr(e.control, 'selected_index') else None
        if tab_index == 1:  # History tab
            self._refresh_history_list()
        elif tab_index == 4:  # Prompt Templates tab
            self._load_prompt_templates_ui()
            try:
                self.page.update()
            except:
                pass
    
    def build(self) -> Container:
        """Build the chat tab with sub-tabs."""
        # Build character browse/edit views
        self._build_char_list()
        self.char_browse_view.controls = [
            Text("🎴 Your AI Characters", size=18, weight=FontWeight.BOLD),
            Text("Click any character to edit it.", size=12, color=Colors.GREY_500),
            Container(height=10),
            self.char_list,
        ]
        
        self.char_edit_view.controls = [
            self.char_back_btn,
            Container(height=10),
            Text("Edit AI Character" if self.editing_character_id else "Create AI Character", size=18, weight=FontWeight.BOLD),
            Container(height=10),
            self.char_name_input,
            self.char_desc_input,
            self.char_system_input,
            Container(height=10),
            Text("Generation Parameters", weight=FontWeight.BOLD),
            Text("Temperature:", size=11),
            self.char_temp_slider,
            self.char_temp_value_text,
            Container(height=8),
            Text("Top P:", size=11),
            self.char_topp_slider,
            self.char_topp_value_text,
            Container(height=8),
            Text("Top K:", size=11),
            self.char_topk_slider,
            self.char_topk_value_text,
            Container(height=8),
            self.char_avatar_input,
            Container(height=10),
            # Prompt template overrides (collapsible)
            ft.ExpansionPanelList(
                elevation=1,
                controls=[
                    ft.ExpansionPanel(
                        header=ft.ListTile(title=Text("📝 Prompt Template Overrides", size=13)),
                        content=Container(
                            content=Column([
                                Text(
                                    "Override the global prompt templates for this character only. "
                                    "Leave fields blank to use the global defaults from the Prompt Templates tab.",
                                    size=11, color=Colors.GREY_500,
                                ),
                                Container(height=6),
                                self.char_override_mode,
                                Container(height=6),
                                self.char_override_char_instructions,
                                self.char_override_memory,
                            ], spacing=4),
                            padding=padding.only(left=12, right=12, bottom=12),
                        ),
                    ),
                ],
            ),
            Container(height=10),
            Row([self.char_save_btn, self.char_delete_btn], spacing=10),
            self.char_status,
        ]
        
        # Build persona browse/edit views
        self._build_persona_list()
        self.persona_browse_view.controls = [
            Text("🎴 Your User Personas", size=18, weight=FontWeight.BOLD),
            Text("Click any persona to edit it.", size=12, color=Colors.GREY_500),
            Container(height=10),
            self.persona_list,
        ]
        
        self.persona_edit_view.controls = [
            self.persona_back_btn,
            Container(height=10),
            Text("Edit User Persona" if self.editing_persona_id else "Create User Persona", size=18, weight=FontWeight.BOLD),
            Container(height=10),
            self.persona_name_input,
            self.persona_desc_input,
            self.persona_bg_input,
            self.persona_avatar_input,
            Container(height=10),
            Row([self.persona_save_btn, self.persona_delete_btn], spacing=10),
            self.persona_status,
        ]
        
        # Build conversation tab content
        conversation_content = Row([
            # Sidebar - Character/Persona & Settings
            Container(
                content=Column([
                    self.model_status,
                    Container(height=10),
                    Text("🎭 Character & Persona", weight=FontWeight.BOLD, size=12),
                    self.char_dropdown,
                    self.persona_dropdown,
                    Container(height=8),
                    self.system_prompt,
                    Container(height=10),
                    Text("⚙️ Parameters", weight=FontWeight.BOLD, size=12),
                    Row([self.temp_slider, self.temp_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                    Row([self.max_tokens_slider, self.max_tokens_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                ], spacing=8, scroll=ScrollMode.AUTO),
                width=280,
                padding=padding.all(15),
                alignment=alignment.top_left,
            ),
            # Main chat area — inner tabs for Chat vs Debug
            Container(
                content=Tabs(
                    tabs=[
                        Tab(
                            text="💬 Chat",
                            content=Container(
                                content=Column([
                                    self.no_model_banner,
                                    Row([
                                        Text("💬 Conversation", size=18, weight=FontWeight.BOLD),
                                        Row([
                                            self.memory_btn,
                                            ElevatedButton(
                                                "➕ New Chat",
                                                on_click=self._new_chat,
                                                height=30,
                                                style=ButtonStyle(
                                                    padding=padding.symmetric(horizontal=8, vertical=2),
                                                    text_style=ft.TextStyle(size=11),
                                                ),
                                            ),
                                            self.clear_btn,
                                        ], spacing=4),
                                    ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                                    self.progress,
                                    self.chat_messages,
                                    # Suggest response button
                                    self.suggest_prompt_btn,
                                    # Suggested response display
                                    self.suggested_prompt_container,
                                    # Typing indicator above message input
                                    self.typing_indicator,
                                    # Directions input (hidden by default, shown when toggled)
                                    self.directions_input,
                                    Row([
                                        self.message_input,
                                        self.directions_toggle,
                                        self.send_btn,
                                    ], spacing=6),
                                ], expand=True),
                                expand=True,
                                padding=padding.only(top=5),
                            ),
                        ),
                        Tab(
                            text="🔍 Debug",
                            content=self.debug_console_content,
                        ),
                    ],
                    expand=True,
                ),
                expand=True,
                padding=padding.all(15),
            ),
        ], expand=True)
        
        # Update initial state
        has_model = app_state.loaded_model_type == "chat"
        has_text = bool(self.message_input.value and self.message_input.value.strip())
        self.send_btn.disabled = not (has_model and has_text)
        self.no_model_banner.visible = not has_model
        
        # Characters tab content
        characters_content = Container(
            content=Column([
                self.char_browse_view,
                self.char_edit_view,
            ], expand=True),
            expand=True,
            padding=padding.all(15),
        )
        
        # Personas tab content
        personas_content = Container(
            content=Column([
                self.persona_browse_view,
                self.persona_edit_view,
            ], expand=True),
            expand=True,
            padding=padding.all(15),
        )
        
        # History tab content
        self._refresh_history_list()
        history_content = Container(
            content=Column([
                Row([
                    Text("📜 Conversation History", size=18, weight=FontWeight.BOLD),
                    self.history_delete_all_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Text("Load, export, or delete prior conversations.", size=12, color=Colors.GREY_500),
                self.history_search,
                self.history_list,
            ], expand=True, spacing=6),
            expand=True,
            padding=padding.all(15),
        )
        
        # Add file picker to page overlay so it can show dialogs
        self.page.overlay.append(self.export_file_picker)
        
        # Prompt templates tab content
        self._load_prompt_templates_ui()
        prompt_templates_content = Container(
            content=Column([
                Text("📝 Chat Prompt Templates", size=18, weight=FontWeight.BOLD),
                Text(
                    "Configure the prompt sections that wrap around every chat message. "
                    "These apply to all conversations. Characters can override individual templates.",
                    size=12, color=Colors.GREY_500,
                ),
                Container(height=10),
                self.pt_char_instructions_input,
                Container(height=6),
                self.pt_persona_input,
                Container(height=6),
                self.pt_directions_input,
                Container(height=6),
                self.pt_memory_input,
                Container(height=10),
                Divider(),
                Container(height=6),
                self.pt_stop_phrases_input,
                self.pt_char_name_stop,
                self.pt_persona_name_stop,
                Container(height=10),
                Row([self.pt_save_btn, self.pt_reset_btn], spacing=10),
                self.pt_status,
            ], expand=True, scroll=ScrollMode.AUTO, spacing=4),
            expand=True,
            padding=padding.all(15),
        )
        
        self._chat_sub_tabs = Tabs(
            tabs=[
                Tab(
                    text="💬 Conversation",
                    content=conversation_content,
                ),
                Tab(
                    text="📜 History",
                    content=history_content,
                ),
                Tab(
                    text="👤 Characters",
                    content=characters_content,
                ),
                Tab(
                    text="🎭 Personas",
                    content=personas_content,
                ),
                Tab(
                    text="📝 Prompts",
                    content=prompt_templates_content,
                ),
            ],
            expand=True,
            on_change=self._on_chat_tab_change,
        )

        return Container(
            content=self._chat_sub_tabs,
            expand=True,
        )


# =============================================================================
# SETTINGS TAB
# =============================================================================
class SettingsTab:
    """Application settings and preferences."""
    
    def __init__(self, page: Page):
        self.page = page
        self._build_ui()
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _build_ui(self):
        from core import CACHE_DIR, GENERATED_IMAGES_DIR, LORA_DIR, CHECKPOINTS_DIR
        
        # Display current paths
        self.cache_path = TextField(
            label="Model Cache Directory",
            value=str(CACHE_DIR),
            read_only=True,
            expand=True,
        )
        self.output_path = TextField(
            label="Generated Images Directory",
            value=str(GENERATED_IMAGES_DIR),
            read_only=True,
            expand=True,
        )
        self.lora_path = TextField(
            label="LoRA Models Directory",
            value=str(LORA_DIR),
            read_only=True,
            expand=True,
        )
        self.checkpoints_path = TextField(
            label="Checkpoints Directory",
            value=str(CHECKPOINTS_DIR),
            read_only=True,
            expand=True,
        )
        
        # Default generation settings with labels
        self.default_steps_value_text = Text("", size=11)
        self.default_steps = Slider(
            min=1, max=50, value=get_setting("default_steps", 25),
            divisions=49,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.default_steps_value_text, "{} steps", int(self.default_steps.value)),
        )
        
        self.default_guidance_value_text = Text("", size=11)
        self.default_guidance = Slider(
            min=1.0, max=20.0, value=get_setting("default_guidance", 7.0),
            divisions=38,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.default_guidance_value_text, "CFG {:.1f}", self.default_guidance.value),
        )
        
        # Initialize settings slider labels
        self._update_slider_label(self.default_steps_value_text, "{} steps", int(self.default_steps.value))
        self._update_slider_label(self.default_guidance_value_text, "CFG {:.1f}", self.default_guidance.value)
        self.default_width = Dropdown(
            label="Default Width",
            width=150,
            options=[
                dropdown.Option("512", "512"),
                dropdown.Option("768", "768"),
                dropdown.Option("1024", "1024"),
            ],
            value=str(get_setting("default_width", 512)),
        )
        self.default_height = Dropdown(
            label="Default Height",
            width=150,
            options=[
                dropdown.Option("512", "512"),
                dropdown.Option("768", "768"),
                dropdown.Option("1024", "1024"),
            ],
            value=str(get_setting("default_height", 512)),
        )
        
        # Save button
        self.save_btn = ElevatedButton(
            "Save Settings",
            icon=Icons.SAVE,
            on_click=self._save_settings,
        )
        
        # Clear cache button
        self.clear_cache_btn = ElevatedButton(
            "Clear Model Cache",
            icon=Icons.DELETE_FOREVER,
            on_click=self._confirm_clear_cache,
            color=Colors.RED_400,
        )
        
        # Open folder buttons
        self.open_cache_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=lambda e: self._open_folder(CACHE_DIR),
        )
        self.open_output_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=lambda e: self._open_folder(GENERATED_IMAGES_DIR),
        )
        self.open_lora_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=lambda e: self._open_folder(LORA_DIR),
        )
        self.open_checkpoints_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=lambda e: self._open_folder(CHECKPOINTS_DIR),
        )

        # Database location (kvg_dblocation, see gerp93/KVG_Standards'
        # db-location-versioning.md) — lets the user relocate chat_history.db
        # instead of it being stuck at a hardcoded path.
        self.db_path_field = TextField(
            label="Chat History Database",
            value=str(chat_db_location.get_effective_db_path()),
            read_only=True,
            expand=True,
        )
        self.db_choose_existing_btn = ElevatedButton(
            "Choose Existing File",
            icon=Icons.FOLDER_OPEN,
            on_click=self._choose_existing_db,
        )
        self.db_choose_new_btn = ElevatedButton(
            "Choose New Location",
            icon=Icons.DRIVE_FILE_MOVE,
            on_click=self._choose_new_db_location,
        )
        self.db_reset_btn = ElevatedButton(
            "Reset to Default",
            icon=Icons.RESTORE,
            on_click=self._reset_db_location,
        )
        self.db_existing_file_picker = ft.FilePicker(on_result=self._on_db_existing_file_picked)
        self.db_new_location_picker = ft.FilePicker(on_result=self._on_db_new_location_picked)
        self.page.overlay.append(self.db_existing_file_picker)
        self.page.overlay.append(self.db_new_location_picker)

        # Updates (kvg_updater bundle mode, see gerp93/KVG_Standards)
        self.version_text = Text(f"Version {CURRENT_VERSION}", size=12, color=Colors.GREY_400)
        self.check_updates_btn = ElevatedButton(
            "Check for Updates",
            icon=Icons.SYSTEM_UPDATE,
            on_click=self._check_for_updates,
        )

    def _open_folder(self, path: Path):
        """Open a folder in the file explorer."""
        import subprocess
        import platform
        
        path_str = str(path)
        if platform.system() == "Windows":
            subprocess.run(["explorer", path_str])
        elif platform.system() == "Darwin":
            subprocess.run(["open", path_str])
        else:
            subprocess.run(["xdg-open", path_str])
    
    def _save_settings(self, e):
        """Save current settings."""
        set_setting("default_steps", int(self.default_steps.value))
        set_setting("default_guidance", self.default_guidance.value)
        set_setting("default_width", int(self.default_width.value))
        set_setting("default_height", int(self.default_height.value))
        
        self.page.snack_bar = SnackBar(content=Text("Settings saved!"))
        self.page.snack_bar.open = True
        self.page.update()

    # ---------------------------------------------------------------
    # Database location (kvg_dblocation) — see gerp93/KVG_Standards'
    # db-location-versioning.md. Relocating always requires a restart:
    # an already-open sqlite3 connection can't be pointed at a new path.
    # ---------------------------------------------------------------

    def _choose_existing_db(self, e):
        """User adopts an existing .db file as-is."""
        self.db_existing_file_picker.pick_files(
            dialog_title="Choose Existing Database File",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["db"],
            allow_multiple=False,
        )

    def _on_db_existing_file_picked(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        self._relocate_db(Path(e.files[0].path))

    def _choose_new_db_location(self, e):
        """User picks a not-yet-existing path; the current DB is copied there."""
        self.db_new_location_picker.save_file(
            dialog_title="Choose New Database Location",
            file_name=chat_db_location.default_filename,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["db"],
        )

    def _on_db_new_location_picked(self, e: ft.FilePickerResultEvent):
        if not e.path:
            return
        self._relocate_db(Path(e.path))

    def _relocate_db(self, new_path: Path):
        try:
            chat_db_location.set_db_path(new_path)
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"Failed to relocate database: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        self.db_path_field.value = str(chat_db_location.get_effective_db_path())
        self._prompt_restart_for_db_change()

    def _reset_db_location(self, e):
        chat_db_location.reset_to_default_db_path()
        self.db_path_field.value = str(chat_db_location.get_effective_db_path())
        self._prompt_restart_for_db_change()

    def _prompt_restart_for_db_change(self):
        def close_dialog(e):
            dialog.open = False
            self.page.update()

        def do_restart(e):
            dialog.open = False
            self.page.update()
            _restart_app()

        dialog = AlertDialog(
            title=Text("Restart Required"),
            content=Text(
                "The chat history database location changed. KVGenius needs "
                "to restart for this to take effect — an already-open "
                "database connection can't be pointed at a new file."
            ),
            actions=[
                TextButton("Restart Later", on_click=close_dialog),
                TextButton("Restart Now", on_click=do_restart),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    # ---------------------------------------------------------------
    # Updates (kvg_updater bundle mode) — see gerp93/KVG_Standards'
    # update-check-versioning.md.
    # ---------------------------------------------------------------

    def _check_for_updates(self, e):
        self.check_updates_btn.disabled = True
        self.page.update()
        threading.Thread(target=self._check_for_updates_worker, args=(True,), daemon=True).start()

    def check_for_updates_silently(self):
        """Startup check: prompts only if an update is actually available."""
        threading.Thread(target=self._check_for_updates_worker, args=(False,), daemon=True).start()

    def _check_for_updates_worker(self, manual: bool):
        try:
            update = check_for_update()
        except Exception as ex:
            update = None
            logger.error(f"Update check failed: {ex}")

        self.check_updates_btn.disabled = False
        if update:
            self._prompt_update(update)
        elif manual:
            self.page.snack_bar = SnackBar(content=Text(f"You're running the latest version ({CURRENT_VERSION})."))
            self.page.snack_bar.open = True
            self.page.update()
        else:
            self.page.update()

    def _prompt_update(self, update: dict):
        def close_dialog(e):
            dialog.open = False
            self.page.update()

        def do_update(e):
            dialog.open = False
            self.page.update()
            threading.Thread(target=self._download_and_apply_update, args=(update,), daemon=True).start()

        dialog = AlertDialog(
            title=Text("Update Available"),
            content=Text(
                f"Version {update['version']} is available (you have {CURRENT_VERSION}).\n\n"
                "Download and install it now? KVGenius will restart automatically."
            ),
            actions=[
                TextButton("Cancel", on_click=close_dialog),
                TextButton("Update Now", on_click=do_update),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _download_and_apply_update(self, update: dict):
        try:
            check_and_apply_update(update)  # never returns on success
        except Exception as ex:
            logger.error(f"Update failed: {ex}")
            self.page.snack_bar = SnackBar(content=Text(f"Update failed: {ex}"), bgcolor=Colors.RED_700)
            self.page.snack_bar.open = True
            self.page.update()

    def _confirm_clear_cache(self, e):
        """Show confirmation dialog before clearing cache."""
        def close_dialog(e):
            dialog.open = False
            self.page.update()
        
        def do_clear(e):
            dialog.open = False
            self._clear_cache()
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Clear Model Cache?"),
            content=Text("This will delete all downloaded models. You'll need to re-download them. Are you sure?"),
            actions=[
                TextButton("Cancel", on_click=close_dialog),
                TextButton("Clear Cache", on_click=do_clear),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _clear_cache(self):
        """Clear the model cache directory."""
        from core import CACHE_DIR
        import shutil
        
        try:
            if CACHE_DIR.exists():
                for item in CACHE_DIR.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            self.page.snack_bar = SnackBar(content=Text("Cache cleared!"))
        except Exception as e:
            self.page.snack_bar = SnackBar(
                content=Text(f"Error: {e}"),
                bgcolor=Colors.RED_700,
            )
        
        self.page.snack_bar.open = True
        self.page.update()
    
    def _get_cache_size(self) -> str:
        """Calculate total size of model cache."""
        from core import CACHE_DIR
        
        total = 0
        if CACHE_DIR.exists():
            for root, dirs, files in os.walk(CACHE_DIR):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except:
                        pass
        
        # Format size
        if total < 1024:
            return f"{total} B"
        elif total < 1024 * 1024:
            return f"{total / 1024:.1f} KB"
        elif total < 1024 * 1024 * 1024:
            return f"{total / (1024 * 1024):.1f} MB"
        else:
            return f"{total / (1024 * 1024 * 1024):.2f} GB"
    
    def build(self) -> Container:
        cache_size = self._get_cache_size()
        
        return Container(
            content=Column([
                Text("⚙️ Settings", size=20, weight=FontWeight.BOLD),
                Container(height=20),
                
                # Paths section
                Text("📁 Directories", size=16, weight=FontWeight.BOLD),
                Container(height=10),
                Row([self.cache_path, self.open_cache_btn]),
                Row([self.output_path, self.open_output_btn]),
                Row([self.lora_path, self.open_lora_btn]),
                Row([self.checkpoints_path, self.open_checkpoints_btn]),
                
                Container(height=20),
                ft.Divider(),
                Container(height=20),
                
                # Default generation settings
                Text("🖌️ Default Generation Settings", size=16, weight=FontWeight.BOLD),
                Container(height=10),
                Text("Steps:", size=11),
                self.default_steps,
                self.default_steps_value_text,
                Container(height=8),
                Text("Guidance Scale:", size=11),
                self.default_guidance,
                self.default_guidance_value_text,
                Container(height=8),
                Row([self.default_width, self.default_height], spacing=20),
                
                Container(height=20),
                self.save_btn,
                
                Container(height=30),
                ft.Divider(),
                Container(height=20),
                
                # Cache management
                Text("🗄️ Cache Management", size=16, weight=FontWeight.BOLD),
                Container(height=10),
                Text(f"Model cache size: {cache_size}", size=12, color=Colors.GREY_400),
                Container(height=10),
                self.clear_cache_btn,

                Container(height=30),
                ft.Divider(),
                Container(height=20),

                # Database location
                Text("🗃️ Database Location", size=16, weight=FontWeight.BOLD),
                Container(height=10),
                Text(
                    "Chat history is stored in a local SQLite file. Relocate it "
                    "(e.g. into a cloud-synced folder) if you'd like — the app "
                    "restarts afterward.",
                    size=11, color=Colors.GREY_400,
                ),
                Container(height=10),
                self.db_path_field,
                Container(height=10),
                Row([self.db_choose_existing_btn, self.db_choose_new_btn, self.db_reset_btn], spacing=10, wrap=True),

                Container(height=30),
                ft.Divider(),
                Container(height=20),

                # Updates
                Text("🔄 Updates", size=16, weight=FontWeight.BOLD),
                Container(height=10),
                self.version_text,
                Container(height=10),
                self.check_updates_btn,
            ], scroll=ScrollMode.AUTO),
            padding=padding.all(20),
            expand=True,
        )


# =============================================================================
# MAIN APP
# =============================================================================
def main(page: Page):
    """Main application entry point."""
    page.title = "KVGenius - AI Image Generator"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window.width = 1600
    page.window.height = 1000
    
    # Apply theme from VisualAssault if available
    if THEMES_AVAILABLE:
        theme = get_theme(app_state.current_theme)
        if theme:
            page.theme = theme
            page.bgcolor = get_theme_background(app_state.current_theme)
            logger.info(f"Applied theme: {app_state.current_theme}")

    # Theme change handler
    def on_theme_change(e):
        if THEMES_AVAILABLE and e.control.value:
            theme_id = e.control.value
            theme = get_theme(theme_id)
            if theme:
                page.theme = theme
                page.bgcolor = get_theme_background(theme_id)
                app_state.current_theme = theme_id
                # Save to settings file for persistence
                set_setting("theme", theme_id)
                page.update()
                logger.info(f"Theme changed to: {theme_id}")
    
    # Create theme selector dropdown
    theme_dropdown = None
    if THEMES_AVAILABLE:
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
    model_manager_tab = ModelManagerTab(page)
    chat_tab = ChatTab(page)
    card_generator_tab = CardGeneratorTab(page)
    
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
    threading.Timer(1.5, settings_tab.check_for_updates_silently).start()

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
                text="📦 Models",
                content=model_manager_tab.build(),
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
