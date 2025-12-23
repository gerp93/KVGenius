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
# GLOBAL STATE
# =============================================================================
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

app_state = AppState()


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
        )
        
        self.send_btn = ElevatedButton(
            "Send",
            icon=Icons.SEND,
            on_click=self._send_message,
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
            tooltip="Clear conversation",
            on_click=self._clear_conversation,
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
        self.send_btn.disabled = not has_model
        self.no_model_banner.visible = not has_model
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
        
        # Clear input and add user message
        self.message_input.value = ""
        self._add_message("user", message)
        self.generating = True
        self.typing_indicator.visible = True
        self.progress.visible = True
        self.send_btn.disabled = True
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
            
            self._add_message("assistant", response)
            self.generating = False
            self.typing_indicator.visible = False
            self.progress.visible = False
            self.send_btn.disabled = False
            
            try:
                self.page.update()
            except:
                pass
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _clear_conversation(self, e):
        """Clear the conversation history."""
        clear_conversation()
        self.chat_messages.controls.clear()
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
                        ft.Icon(Icons.EDIT, size=18, color=Colors.GREY_400),
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
                        ft.Icon(Icons.EDIT, size=18, color=Colors.GREY_400),
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
                        self.clear_btn,
                    ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                    self.progress,
                    self.chat_messages,
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
        self.send_btn.disabled = not has_model
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
        
        return Container(
            content=Tabs(
                tabs=[
                    Tab(
                        text="💬 Conversation",
                        content=conversation_content,
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
    page.window.width = 1200
    page.window.height = 900
    
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
