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
    Page, Text, Column, Row, Container, Card, Tabs, Tab, TabBar, TabBarView,
    TextField, ElevatedButton, ProgressBar, Image, Dropdown,
    Slider, IconButton, ListView, Colors, Icons,
    MainAxisAlignment, CrossAxisAlignment, ScrollMode,
    dropdown, SnackBar, AlertDialog, TextButton, FontWeight,
    padding, border_radius, border, GridView, Checkbox, Stack, alignment,
    ControlState, ButtonStyle, Divider,
)
alignment = alignment.Alignment  # flet 0.86.5: Alignment constants replace old module attrs

padding = padding.Padding  # flet 0.86.5: Padding classmethods replace old module functions
border_radius = border_radius.BorderRadius  # flet 0.86.5: BorderRadius classmethods replace old module functions
border = border.Border  # flet 0.86.5: Border classmethods replace old module functions


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
    # Chat (only used to keep a model loaded for the Card Generator)
    get_available_chat_models,
    is_chat_model_downloaded,
    _is_chat_model_fully_downloaded,
    load_chat_model,
    unload_chat_model,
    download_chat_model,
)

# Import CardGeneratorTab from ui.tabs
from ui.tabs.card_generator import CardGeneratorTab

# Self-update (kvg_updater bundle mode, see gerp93/KVG_Standards)
from updater import CURRENT_VERSION, check_for_update, check_and_apply_update


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
            on_select=self._on_model_change,
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
            on_select=self._on_layout_change,
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
            on_select=self._on_dimension_change,
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
            on_select=self._on_dimension_change,
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
            fit=ft.BoxFit.CONTAIN,
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
            self.page.show_dialog(SnackBar(content=Text("Please enter a prompt")))
            self.page.update()
            return
        
        # Check if model is loaded
        current_model, current_key = get_current_model()
        debug_print(f" Model state: exists={current_model is not None}, key={current_key}, app_state={app_state.image_model_loaded}")
        
        if current_model is None:
            print("[DEBUG] No model loaded - showing error")
            self.page.show_dialog(SnackBar(
                content=Text("❌ No model loaded! Load a model first."),
                bgcolor=Colors.RED_900,
            ))
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
                    self.page.show_dialog(SnackBar(
                        content=Text("❌ Generation failed - check terminal for details"),
                        bgcolor=Colors.RED_900,
                    ))
                
                self.page.update()
                
            except Exception as ex:
                logger.error(f"Generation exception: {ex}", exc_info=True)
                app_state.is_generating = False
                self.generate_btn.disabled = False
                self.generate_btn.text = "🎨 Generate"
                self.stop_btn.disabled = True
                self.progress_bar.value = 0
                self.progress_text.value = ""
                
                self.page.show_dialog(SnackBar(
                    content=Text(f"❌ Error: {str(ex)[:100]}"),
                    bgcolor=Colors.RED_900,
                ))
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
            self.page.show_dialog(SnackBar(
                content=Text("Enter a prompt first!"),
                bgcolor=Colors.ORANGE_700,
            ))
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
            self.page.show_dialog(SnackBar(
                content=Text("Prompt library not connected."),
                bgcolor=Colors.RED_700,
            ))
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
            alignment=ft.Alignment.CENTER,
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
                alignment=ft.Alignment.CENTER,
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
                alignment=ft.Alignment.CENTER,
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
            on_select=self._on_page_size_change,
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
            fit=ft.BoxFit.CONTAIN,
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
            self.page.show_dialog(SnackBar(content=Text(f"🗑️ Deleted {deleted} images")))
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
            
            self.page.show_dialog(SnackBar(
                content=Text(f"✅ Downloaded {downloaded} images to {downloads_dir}")
            ))
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
        self.page.show_dialog(SnackBar(content=Text("📋 Prompt copied to clipboard!")))
        self.page.update()
    
    def _delete_image(self, filepath: str):
        """Delete an image with confirmation."""
        def do_delete(e):
            try:
                Path(filepath).unlink()
                self.selected_images.discard(filepath)
                self._refresh()
                self.page.show_dialog(SnackBar(content=Text("🗑️ Image deleted")))
            except Exception as ex:
                self.page.show_dialog(SnackBar(content=Text(f"❌ Error: {ex}")))
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
                                fit=ft.BoxFit.COVER,
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
                    alignment=alignment.CENTER,
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
                                fit=ft.BoxFit.COVER,
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
                            alignment=alignment.TOP_LEFT,
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
                            alignment=alignment.BOTTOM_CENTER,
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
                    alignment=alignment.CENTER,
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
            length=3,
            selected_index=0,
            animation_duration=300,
            expand=True,
            content=Column([
                TabBar(
                    tabs=[
                        Tab(label="🎨 Image Models"),
                        Tab(label="💬 Chat / LLM Models"),
                        Tab(label="📥 Downloads"),
                    ],
                ),
                TabBarView(
                    expand=True,
                    controls=[
                        self.image_model_list,
                        self.chat_model_list,
                        self.downloads_list,
                    ],
                ),
            ], expand=True),
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
        self.page.show_dialog(SnackBar(
            content=Text("✅ Prompt loaded! Switched to Generate tab."),
        ))
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
                self.page.show_dialog(SnackBar(content=Text(f"✂️ Cloned: {clone['name']}")))
                self._filter_and_display_prompts()
            else:
                self.page.show_dialog(SnackBar(
                    content=Text("Failed to clone prompt."),
                    bgcolor=Colors.RED_700,
                ))
                self.page.update()
    
    def _copy_prompt(self, prompt: str):
        """Copy prompt to clipboard."""
        self.page.set_clipboard(prompt)
        self.page.show_dialog(SnackBar(content=Text("Prompt copied!")))
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
            self.page.show_dialog(SnackBar(content=Text(f"✅ Saved: {name}")))
            self._load_prompts()
        else:
            self.page.show_dialog(SnackBar(
                content=Text("Failed to save prompt."),
                bgcolor=Colors.RED_700,
            ))
            self.page.update()
    
    def _delete_prompt(self, index: int):
        """Delete a saved prompt."""
        data = self._load_saved_prompts()
        if 0 <= index < len(data["prompts"]):
            deleted = data["prompts"].pop(index)
            self._save_prompts_file(data)
            self.page.show_dialog(SnackBar(content=Text(f"Deleted: {deleted.get('name', 'prompt')}")))
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
            self.page.show_dialog(SnackBar(content=Text(f"💾 Saved to library: {name[:30]}...")))
            self._load_prompts()
        else:
            self.page.show_dialog(SnackBar(
                content=Text("Failed to save prompt."),
                bgcolor=Colors.RED_700,
            ))
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
        
        self.page.show_dialog(SnackBar(content=Text("Settings saved!")))
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
            self.page.show_dialog(SnackBar(content=Text(f"You're running the latest version ({CURRENT_VERSION}).")))
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
            self.page.show_dialog(SnackBar(content=Text(f"Update failed: {ex}"), bgcolor=Colors.RED_700))
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
            
            self.page.show_dialog(SnackBar(content=Text("Cache cleared!")))
        except Exception as e:
            self.page.show_dialog(SnackBar(
                content=Text(f"Error: {e}"),
                bgcolor=Colors.RED_700,
            ))
        
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
            on_select=on_theme_change,
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
    
    # Settings tab
    settings_tab = SettingsTab(page)
    threading.Timer(1.5, settings_tab.check_for_updates_silently).start()

    # Create header
    header = HeaderBar(page, on_model_loaded=on_model_loaded, theme_dropdown=theme_dropdown)
    
    # Build tabs
    tabs = Tabs(
        length=6,
        selected_index=0,
        animation_duration=200,
        expand=True,
        content=Column([
            TabBar(
                tabs=[
                    Tab(label="🖌️ Generate"),
                    Tab(label="🃏 Cards"),
                    Tab(label="📚 Prompts"),
                    Tab(label="🖼️ Gallery"),
                    Tab(label="📦 Models"),
                    Tab(label="⚙️ Settings"),
                ],
            ),
            TabBarView(
                expand=True,
                controls=[
                    image_gen_tab.build(),
                    card_generator_tab.build(),
                    prompt_library_tab.build(),
                    gallery_tab.build(),
                    model_manager_tab.build(),
                    settings_tab.build(),
                ],
            ),
        ], expand=True),
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
