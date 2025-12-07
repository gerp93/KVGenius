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
from typing import Optional, Dict, Any

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
    padding, border_radius, border
)

# Import KVG Themes
try:
    from flet_kvg_themes import get_theme, get_theme_list, theme_exists
    THEMES_AVAILABLE = True
except ImportError:
    THEMES_AVAILABLE = False
    logger.warning("flet_kvg_themes not installed - using default theme")

# Import core engine
from core import (
    get_available_image_models,
    is_model_downloaded,
    load_image_model,
    unload_image_model,
    generate_image,
    get_current_model,
    get_generated_images,
    GenerationResult,
    HUGGINGFACE_MODELS,
)

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
        self.image_model_loaded: Optional[str] = None
        self.chat_model_loaded: Optional[str] = None
        self.current_theme: str = "dark"  # Default theme
        self.is_generating: bool = False
        self.cancel_requested: bool = False

app_state = AppState()


# =============================================================================
# HELPER FUNCTIONS - Match Gradio's logic
# =============================================================================
def get_image_model_choices():
    """Get image model choices with status indicators (matches Gradio)."""
    models = get_available_image_models()
    choices = []
    
    for name, info in models.items():
        # Determine status
        is_loaded = (app_state.image_model_loaded == name)
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


def get_image_model_info(model_name: str) -> str:
    """Get detailed info for a model (matches Gradio's get_image_model_info)."""
    models = get_available_image_models()
    if model_name not in models:
        return f"Unknown model: {model_name}"
    
    info = models[model_name]
    is_loaded = (app_state.image_model_loaded == model_name)
    is_downloaded = info.get("source") == "local" or is_model_downloaded(info.get("id", ""))
    
    status = "🟢 LOADED" if is_loaded else ("🟡 Downloaded" if is_downloaded else "🔴 Not Downloaded")
    
    lines = [
        f"🎨 {model_name} | {status}",
        f"Type: {info.get('type', 'sd').upper()} | VRAM: {info.get('vram', 'Unknown')}",
    ]
    
    if info.get("description"):
        lines.append(info['description'])
    
    return " | ".join(lines)


# =============================================================================
# HEADER BAR - Global Model Selectors (matches Gradio header)
# =============================================================================
class HeaderBar:
    """Global header with model selectors - matches Gradio layout."""
    
    def __init__(self, page: Page, on_model_loaded: callable = None, theme_dropdown: Dropdown = None):
        self.page = page
        self.on_model_loaded = on_model_loaded
        self.theme_dropdown = theme_dropdown
        self._build_ui()
    
    def _build_ui(self):
        # Image Model Dropdown
        choices = get_image_model_choices()
        self.img_dropdown = Dropdown(
            label="🎨 Image Model",
            width=300,
            options=[dropdown.Option(key=c["key"], text=c["text"]) for c in choices],
            value=choices[0]["key"] if choices else None,
            on_change=self._on_img_model_change,
        )
        
        # Load/Unload buttons (toggle visibility like Gradio)
        self.img_load_btn = ElevatedButton(
            "📥 Load",
            on_click=self._load_image_model,
            bgcolor=Colors.BLUE_700,
            color=Colors.WHITE,
        )
        self.img_unload_btn = ElevatedButton(
            "🗑️ Unload",
            on_click=self._unload_image_model,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            visible=False,  # Hidden until model loaded
        )
        self.img_refresh_btn = IconButton(
            icon=Icons.REFRESH,
            on_click=self._refresh_img_models,
            tooltip="Refresh model list",
        )
        
        # Status bar
        self.status_text = Text(
            "💡 Select a model to see details, then click Load to start",
            size=13,
            color=Colors.GREY_400,
        )
    
    def _on_img_model_change(self, e):
        """Update status when model selection changes."""
        model_key = self.img_dropdown.value
        if model_key:
            info = get_image_model_info(model_key)
            self.status_text.value = info
            self.page.update()
    
    def _refresh_img_models(self, e):
        """Refresh the model dropdown."""
        choices = get_image_model_choices()
        self.img_dropdown.options = [dropdown.Option(key=c["key"], text=c["text"]) for c in choices]
        self.page.update()
    
    def _update_button_visibility(self):
        """Toggle Load/Unload button visibility based on state (matches Gradio)."""
        if app_state.image_model_loaded:
            self.img_load_btn.visible = False
            self.img_unload_btn.visible = True
        else:
            self.img_load_btn.visible = True
            self.img_unload_btn.visible = False
    
    def _load_image_model(self, e):
        """Load the selected image model."""
        model_key = self.img_dropdown.value
        if not model_key:
            return
        
        self.img_load_btn.disabled = True
        self.status_text.value = f"⏳ Loading {model_key}..."
        self.page.update()
        
        def do_load():
            def progress_cb(pct, msg):
                self.status_text.value = f"⏳ [{pct*100:.0f}%] {msg}"
                self.page.update()
            
            success, message = load_image_model(model_key, progress_callback=progress_cb)
            
            # Verify model actually loaded
            current_model, current_key = get_current_model()
            debug_print(f" Load result: success={success}, message={message}")
            debug_print(f" After load: current_model exists={current_model is not None}, current_key={current_key}")
            
            if success:
                app_state.image_model_loaded = model_key
                self.status_text.value = f"🟢 {model_key} loaded and ready!"
            else:
                self.status_text.value = f"❌ Failed: {message}"
            
            self.img_load_btn.disabled = False
            self._update_button_visibility()
            self._refresh_img_models(None)  # Update dropdown status indicators
            
            # Notify tab that model is loaded
            if self.on_model_loaded:
                self.on_model_loaded(success, model_key)
            
            self.page.update()
        
        threading.Thread(target=do_load, daemon=True).start()
    
    def _unload_image_model(self, e):
        """Unload the current image model."""
        model_key = app_state.image_model_loaded
        if not model_key:
            return
        
        self.status_text.value = f"🗑️ Unloading {model_key}..."
        self.page.update()
        
        unload_image_model()
        app_state.image_model_loaded = None
        
        self.status_text.value = "💡 Model unloaded. Select a model to load."
        self._update_button_visibility()
        self._refresh_img_models(None)
        
        if self.on_model_loaded:
            self.on_model_loaded(False, None)
        
        self.page.update()
    
    def build(self) -> Container:
        """Build the header bar."""
        # Build the model selector row
        model_row_items = [
            self.img_dropdown,
            self.img_load_btn,
            self.img_unload_btn,
            self.img_refresh_btn,
        ]
        
        # Add theme dropdown if available
        if self.theme_dropdown:
            model_row_items.insert(0, self.theme_dropdown)
            model_row_items.insert(1, Container(width=20))  # Spacer
        
        return Container(
            content=Column([
                # Top row: Title + Model selectors + Theme
                Row([
                    # Title
                    Column([
                        Text("🤖 KVGenius AI Studio", size=22, weight=FontWeight.BOLD),
                        Text("GPU-Powered Local Generation", size=11, color=Colors.GREY_500),
                    ], spacing=2),
                    
                    # Spacer
                    Container(expand=True),
                    
                    # Theme + Image Model selector with Load/Unload buttons
                    Row(model_row_items, spacing=8),
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                
                # Status bar
                Container(
                    content=self.status_text,
                    padding=padding.only(top=8),
                ),
            ]),
            padding=padding.all(15),
            # Use theme's surface color (don't hardcode)
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
    
    def __init__(self, page: Page):
        self.page = page
        self.current_result: Optional[GenerationResult] = None
        self.current_layout = self.LAYOUT_AUTO
        self._build_ui()
    
    def set_generate_enabled(self, enabled: bool):
        """Enable/disable generate button based on model state."""
        self.generate_btn.disabled = not enabled
        self.page.update()
    
    def _get_optimal_layout(self) -> str:
        """Determine optimal layout based on aspect ratio."""
        if self.current_layout != self.LAYOUT_AUTO:
            return self.current_layout
        
        # Get current dimensions
        width = int(self.width_slider.value)
        height = int(self.height_slider.value)
        
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
        self.lora_dropdown = Dropdown(
            label="LoRA",
            options=[dropdown.Option("None")],
            value="None",
            width=160,
        )
        self.lora_strength = Slider(
            min=0.0, max=1.5, divisions=30, value=0.8,
            label="Strength: {value}",
            width=160,
        )
        
        self.steps_slider = Slider(
            min=1, max=50, divisions=49, value=4,
            label="Steps: {value}",
            width=160,
        )
        self.guidance_slider = Slider(
            min=0, max=20, divisions=40, value=0,
            label="Guidance: {value}",
            width=160,
        )
        self.width_slider = Slider(
            min=512, max=1024, divisions=8, value=512,
            label="Width: {value}",
            width=160,
            on_change=self._on_dimension_change,
        )
        self.height_slider = Slider(
            min=512, max=1024, divisions=8, value=512,
            label="Height: {value}",
            width=160,
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
            bgcolor=Colors.BLUE_700,
            color=Colors.WHITE,
            disabled=True,
            height=45,
        )
        self.stop_btn = ElevatedButton(
            "⏹ Stop",
            on_click=self._stop_generation,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            disabled=True,
            height=45,
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
        
        # Main container that will hold the dynamic layout
        self.main_container = Container(expand=True)
    
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
                    width=int(self.width_slider.value),
                    height=int(self.height_slider.value),
                    seed=int(self.seed_field.value or -1),
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
    
    def _build_settings_panel(self) -> Container:
        """Build the compact settings panel."""
        return Container(
            content=Column([
                Row([self.layout_dropdown], alignment=MainAxisAlignment.START),
                Container(height=10),
                Text("🎭 LoRA", weight=FontWeight.BOLD, size=13),
                self.lora_dropdown,
                self.lora_strength,
                Container(height=10),
                Text("⚙️ Settings", weight=FontWeight.BOLD, size=13),
                self.steps_slider,
                self.guidance_slider,
                self.width_slider,
                self.height_slider,
                self.seed_field,
            ], spacing=5, scroll=ScrollMode.AUTO),
            width=200,
            padding=padding.all(12),
            bgcolor=Colors.with_opacity(0.15, Colors.ON_SURFACE),
            border_radius=border_radius.all(8),
        )
    
    def _build_prompt_section(self) -> Column:
        """Build the prompts and generate buttons section."""
        return Column([
            self.prompt_field,
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
        img_width = int(self.width_slider.value)
        img_height = int(self.height_slider.value)
        
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
                self.lora_strength,
                Container(height=10),
                Text("⚙️ Settings", weight=FontWeight.BOLD, size=13),
                self.steps_slider,
                self.guidance_slider,
                self.width_slider,
                self.height_slider,
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
        
        return Container(
            content=self.main_container,
            padding=padding.all(15),
            expand=True,
        )


# =============================================================================
# GALLERY TAB
# =============================================================================
class GalleryTab:
    """Gallery Tab - view generated images."""
    
    def __init__(self, page: Page):
        self.page = page
        self._build_ui()
    
    def _build_ui(self):
        self.image_list = ListView(spacing=10, padding=20, expand=True)
        self.refresh_btn = ElevatedButton(
            "🔄 Refresh",
            on_click=self._refresh,
        )
    
    def _refresh(self, e=None):
        self._load_images()
        self.page.update()
    
    def _load_images(self):
        images = get_generated_images(limit=50)
        self.image_list.controls.clear()
        
        if not images:
            self.image_list.controls.append(
                Text("No generated images yet", color=Colors.GREY_500)
            )
            return
        
        for img_info in images:
            card = Card(
                content=Container(
                    content=Row([
                        Image(
                            src=img_info["path"],
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
                        ], expand=True, spacing=3),
                    ], spacing=15),
                    padding=padding.all(10),
                ),
            )
            self.image_list.controls.append(card)
    
    def build(self) -> Container:
        self._load_images()
        return Container(
            content=Column([
                Row([
                    Text("🖼️ Generated Images", size=18, weight=FontWeight.BOLD),
                    self.refresh_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                self.image_list,
            ]),
            padding=padding.all(15),
            expand=True,
        )


# =============================================================================
# MODEL MANAGER TAB
# =============================================================================
class ModelManagerTab:
    """Model Manager - view and download models."""
    
    def __init__(self, page: Page):
        self.page = page
        self._build_ui()
    
    def _build_ui(self):
        self.model_list = ListView(spacing=10, padding=20, expand=True)
    
    def _load_models(self):
        models = get_available_image_models()
        self.model_list.controls.clear()
        
        for name, info in models.items():
            is_hf = info.get("source") == "huggingface"
            downloaded = is_model_downloaded(info.get("id", "")) if is_hf else True
            is_loaded = (app_state.image_model_loaded == name)
            
            if is_loaded:
                status_icon = Icons.CHECK_CIRCLE
                status_color = Colors.GREEN_400
                status_text = "Loaded"
            elif downloaded:
                status_icon = Icons.DOWNLOAD_DONE
                status_color = Colors.YELLOW_400
                status_text = "Downloaded"
            else:
                status_icon = Icons.CLOUD_DOWNLOAD
                status_color = Colors.GREY_500
                status_text = "Not Downloaded"
            
            card = Card(
                content=Container(
                    content=Row([
                        ft.Icon(status_icon, color=status_color, size=28),
                        Column([
                            Text(name, weight=FontWeight.BOLD),
                            Text(info.get("description", ""), size=11, color=Colors.GREY_400),
                            Row([
                                Container(
                                    content=Text(info.get("type", "sd").upper(), size=10),
                                    bgcolor=Colors.BLUE_900,
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
                    ], spacing=15),
                    padding=padding.all(12),
                ),
            )
            self.model_list.controls.append(card)
    
    def build(self) -> Container:
        self._load_models()
        return Container(
            content=Column([
                Text("📦 Model Manager", size=18, weight=FontWeight.BOLD),
                Text("🟢 Loaded | 🟡 Downloaded | 🔴 Not Downloaded", size=11, color=Colors.GREY_500),
                Container(height=10),
                self.model_list,
            ]),
            padding=padding.all(15),
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
                page.update()
                logger.info(f"Theme changed to: {theme_id}")
    
    # Create theme selector dropdown
    theme_dropdown = None
    if THEMES_AVAILABLE:
        themes = get_theme_list()
        theme_dropdown = Dropdown(
            label="🎨 Theme",
            width=180,
            options=[dropdown.Option(key=tid, text=name) for tid, name in themes],
            value=app_state.current_theme,
            on_change=on_theme_change,
        )
    
    # Create tabs
    image_gen_tab = ImageGenTab(page)
    gallery_tab = GalleryTab(page)
    model_manager_tab = ModelManagerTab(page)
    
    # Callback when model loads/unloads
    def on_model_loaded(success: bool, model_key: Optional[str]):
        image_gen_tab.set_generate_enabled(success and model_key is not None)
    
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
                text="🖼️ Gallery",
                content=gallery_tab.build(),
            ),
            Tab(
                text="📦 Models",
                content=model_manager_tab.build(),
            ),
        ],
        expand=True,
    )
    
    page.add(
        Column([
            header.build(),
            tabs,
        ], expand=True, spacing=0)
    )


if __name__ == "__main__":
    ft.app(target=main)
