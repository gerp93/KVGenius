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

# Import KVG Themes
try:
    from flet_kvg_themes import get_theme, get_theme_list, theme_exists
    THEMES_AVAILABLE = True
except ImportError:
    THEMES_AVAILABLE = False
    print("Warning: flet_kvg_themes not installed - using default theme")

# Import core engine
from core import (
    get_available_image_models,
    is_model_downloaded,
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
        self._build_ui()
    
    def _build_ui(self):
        self.model_list = ListView(spacing=10, padding=20, expand=True)
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh model list",
            on_click=lambda e: self._refresh(),
        )
        self.status_text = Text("", size=11, color=Colors.GREY_400)
    
    def _refresh(self):
        """Refresh the model list."""
        self._load_models()
        if self.page:
            self.page.update()
    
    def _download_model(self, model_name: str, repo_id: str, model_type: str):
        """Start downloading a model in background."""
        if repo_id in self._downloading and self._downloading[repo_id]:
            return  # Already downloading
        
        self._downloading[repo_id] = True
        self.status_text.value = f"Downloading {model_name}..."
        self.page.update()
        
        def do_download():
            def progress_callback(pct: float, msg: str):
                self.status_text.value = f"[{pct*100:.0f}%] {msg}"
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
            
            if success:
                self.status_text.value = f"✅ {model_name} downloaded!"
            else:
                self.status_text.value = f"❌ {message}"
            
            # Refresh the list
            self._load_models()
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=do_download, daemon=True).start()
    
    def _load_models(self):
        self.model_list.controls.clear()
        
        # Add section header for Image Models
        self.model_list.controls.append(
            Container(
                content=Text("🎨 Image Generation Models", size=14, weight=FontWeight.BOLD),
                padding=padding.only(bottom=5, top=10),
            )
        )
        
        # Image Models
        image_models = get_available_image_models()
        for name, info in image_models.items():
            is_hf = info.get("source") == "huggingface"
            repo_id = info.get("id", "")
            downloaded = is_model_downloaded(repo_id) if is_hf else True
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "image")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(name, info, "image", is_loaded, downloaded, is_downloading, is_hf, repo_id)
        
        # Add section header for Chat Models
        self.model_list.controls.append(
            Container(
                content=Text("💬 Chat / LLM Models", size=14, weight=FontWeight.BOLD),
                padding=padding.only(bottom=5, top=20),
            )
        )
        
        # Chat Models
        chat_models = get_available_chat_models()
        for name, info in chat_models.items():
            repo_id = info.get("id", "")
            downloaded = is_chat_model_downloaded(repo_id)
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "chat")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(name, info, "chat", is_loaded, downloaded, is_downloading, True, repo_id)
    
    def _add_model_card(self, name: str, info: Dict, model_type: str, is_loaded: bool, 
                        downloaded: bool, is_downloading: bool, is_hf: bool, repo_id: str):
        """Add a model card to the list."""
        if is_loaded:
            status_icon = Icons.CHECK_CIRCLE
            status_color = Colors.GREEN_400
            status_text = "Loaded"
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
        
        if is_hf and not downloaded and not is_downloading:
            action_buttons.append(
                ElevatedButton(
                    "Download",
                    icon=Icons.DOWNLOAD,
                    on_click=lambda e, n=name, r=repo_id, t=model_type: self._download_model(n, r, t),
                )
            )
        elif is_downloading:
            action_buttons.append(
                Row([
                    ft.ProgressRing(width=16, height=16, stroke_width=2),
                    Text("Downloading...", size=11),
                ], spacing=5)
            )
        
        # Type badge color
        type_text = info.get("type", "sd").upper() if model_type == "image" else "LLM"
        type_color = Colors.BLUE_900 if model_type == "image" else Colors.ORANGE_900
        
        card = Card(
            content=Container(
                content=Row([
                    ft.Icon(status_icon, color=status_color, size=28),
                    Column([
                        Text(name, weight=FontWeight.BOLD),
                        Text(info.get("description", ""), size=11, color=Colors.GREY_400),
                        Row([
                            Container(
                                content=Text(type_text, size=10),
                                bgcolor=type_color,
                                padding=padding.all(5),
                                border_radius=border_radius.all(5),
                            ),
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
                        ], spacing=5),
                    ], expand=True, spacing=3),
                    Column(action_buttons, spacing=5) if action_buttons else Container(),
                ], spacing=15),
                padding=padding.all(12),
            ),
        )
        self.model_list.controls.append(card)
    
    def build(self) -> Container:
        self._load_models()
        return Container(
            content=Column([
                Row([
                    Text("📦 Model Manager", size=18, weight=FontWeight.BOLD),
                    self.refresh_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Row([
                    Text("🟢 Loaded | 🟡 Downloaded | ⚪ Not Downloaded", size=11, color=Colors.GREY_500),
                    self.status_text,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Container(height=10),
                self.model_list,
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
        self.db = ChatHistoryDB(db_path="./data/chat_history.db")
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
            label="Type your message... (Enter to send, Shift+Enter for newline)",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
            shift_enter=True,
            on_submit=self._send_message,
            on_change=self._on_message_input_change,
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
        
        self.clear_btn = IconButton(
            icon=Icons.DELETE_SWEEP,
            tooltip="Delete conversation",
            on_click=self._delete_current_conversation,
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
            expand=True,
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
        
        # Browse vs Edit views
        self.char_browse_view = Column([], expand=True)
        self.char_edit_view = Column([], expand=True, visible=False)
        
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
        
        bubble = Container(
            content=Column([
                Text(
                    display_name,
                    size=10,
                    weight=FontWeight.BOLD,
                    color=name_color,
                ),
                Text(content, selectable=True),
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
            # Build system prompt from character or manual input
            if self.selected_character:
                system = self.selected_character.get('system_prompt', '')
                temp = self.selected_character.get('temperature', 0.7)
            else:
                system = self.system_prompt.value or None
                temp = self.temp_slider.value
            
            # Append persona background if selected
            if self.selected_persona and self.selected_persona.get('background'):
                persona_context = f"\n\n[User is roleplaying as: {self.selected_persona['name']}]\n{self.selected_persona['background']}"
                if system:
                    system = system + persona_context
                else:
                    system = persona_context
            
            success, response = generate_chat_response(
                user_message=message,
                system_prompt=system,
                max_new_tokens=int(self.max_tokens_slider.value),
                temperature=temp,
            )
            
            # generate_chat_response added user+assistant to _conversation_history
            # Pop the assistant - it stays uncommitted until user sends next message
            import core.chat_gen as _chat_mod
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "assistant":
                _chat_mod._conversation_history.pop()
            
            # Initialize regen carousel with this response (uncommitted)
            self.regen_responses = [response]
            self.regen_index = 0
            self.response_committed = False
            
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
        self._update_dropdown_lock()
        self.page.update()
    
    def _new_chat(self, e=None):
        """Start a new chat - commits pending response if any, then clears."""
        self._commit_pending_response()
        self._clear_conversation()
    
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
        self.regen_text_ctrl.value = self.regen_responses[self.regen_index]
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
            
            # Pop user from history (generate_chat_response will re-add it)
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "user":
                _chat_mod._conversation_history.pop()
            
            if self.selected_character:
                system = self.selected_character.get('system_prompt', '')
                temp = self.selected_character.get('temperature', 0.7)
            else:
                system = self.system_prompt.value or None
                temp = self.temp_slider.value
            
            if self.selected_persona and self.selected_persona.get('background'):
                persona_context = f"\n\n[User is roleplaying as: {self.selected_persona['name']}]\n{self.selected_persona['background']}"
                if system:
                    system = system + persona_context
                else:
                    system = persona_context
            
            success, response = generate_chat_response(
                user_message=self.last_user_message,
                system_prompt=system,
                max_new_tokens=int(self.max_tokens_slider.value),
                temperature=temp,
            )
            
            # Pop assistant from history (keep user)
            if _chat_mod._conversation_history and _chat_mod._conversation_history[-1].get("role") == "assistant":
                _chat_mod._conversation_history.pop()
            
            # Add to carousel and show latest
            self.regen_responses.append(response)
            self.regen_index = len(self.regen_responses) - 1
            
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
    
    def _on_chat_tab_change(self, e):
        """Handle sub-tab changes - refresh History when selected."""
        tab_index = e.control.selected_index if hasattr(e.control, 'selected_index') else None
        if tab_index == 1:  # History tab
            self._refresh_history_list()
    
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
            # Main chat area
            Container(
                content=Column([
                    self.no_model_banner,
                    Row([
                        Text("💬 Conversation", size=18, weight=FontWeight.BOLD),
                        Row([
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
                    Row([
                        self.message_input,
                        self.send_btn,
                    ], spacing=10),
                ], expand=True),
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
        
        return Container(
            content=Tabs(
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
                ],
                expand=True,
                on_change=self._on_chat_tab_change,
            ),
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
    
    # Apply theme from KVG Themes if available
    if THEMES_AVAILABLE:
        theme = get_theme(app_state.current_theme)
        if theme:
            page.theme = theme
            logger.info(f"Applied theme: {app_state.current_theme}")
    
    # Theme change handler
    def on_theme_change(e):
        if THEMES_AVAILABLE and e.control.value:
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
