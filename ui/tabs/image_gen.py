"""
Image Generation Tab for KVGenius Desktop App.
"""
import threading
import logging
from typing import Optional, Callable
from pathlib import Path

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, TextField, ElevatedButton, 
    ProgressBar, Image, Dropdown, Slider, IconButton, Colors, Icons,
    MainAxisAlignment, CrossAxisAlignment, ScrollMode, dropdown, SnackBar,
    FontWeight, padding, border_radius, ControlState, ButtonStyle,
)

from ui.state import app_state
from core import (
    generate_image,
    get_current_model,
    get_available_loras,
    GenerationResult,
)

logger = logging.getLogger(__name__)


def debug_print(msg):
    """Print with immediate flush."""
    import sys
    sys.stdout.write(f"[DEBUG] {msg}\n")
    sys.stdout.flush()


class ImageGenTab:
    """Image Generation Tab with dynamic image placement."""
    
    # Layout modes
    LAYOUT_AUTO = "Auto (by aspect ratio)"
    LAYOUT_RIGHT = "Image on Right"
    LAYOUT_BOTTOM = "Image on Bottom"
    
    def __init__(self, page: Page, on_save_prompt: Optional[Callable] = None):
        self.page = page
        self.on_save_prompt = on_save_prompt
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
        self._refresh_lora_dropdown()
        try:
            self.page.update()
        except:
            pass
    
    def _refresh_lora_dropdown(self):
        """Refresh LoRA dropdown filtered by loaded model type."""
        model_type = app_state.image_model_arch
        available_loras = get_available_loras(model_type)
        
        lora_options = [dropdown.Option("None")]
        for lora in available_loras:
            lora_options.append(dropdown.Option(lora["name"]))
        
        current_value = self.lora_dropdown.value
        self.lora_dropdown.options = lora_options
        
        valid_names = [opt.key for opt in lora_options]
        if current_value not in valid_names:
            self.lora_dropdown.value = "None"
    
    def _get_optimal_layout(self) -> str:
        """Determine optimal layout based on aspect ratio."""
        if self.current_layout != self.LAYOUT_AUTO:
            return self.current_layout
        
        width = int(self.width_dropdown.value)
        height = int(self.height_dropdown.value)
        
        if height >= width:
            return self.LAYOUT_RIGHT
        else:
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
        
        if layout == self.LAYOUT_RIGHT:
            self.main_container.content = self._build_side_layout()
        else:
            self.main_container.content = self._build_bottom_layout()
        
        self.page.update()
    
    def _build_ui(self):
        """Build UI components."""
        
        # Layout selector
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
        
        # LoRA settings
        lora_options = [dropdown.Option("None")]
        for lora in get_available_loras():
            lora_options.append(dropdown.Option(lora["name"]))
        
        self.lora_dropdown = Dropdown(
            label="LoRA",
            options=lora_options,
            value="None",
            width=160,
        )
        
        # Sliders
        self.lora_strength_value_text = Text("", size=11)
        self.lora_strength = Slider(
            min=0.0, max=1.5, divisions=30, value=0.8,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.lora_strength_value_text, "LoRA {:.2f}", self.lora_strength.value),
        )
        
        self.steps_value_text = Text("", size=11)
        self.steps_slider = Slider(
            min=1, max=50, divisions=49, value=25,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.steps_value_text, "{} steps", int(self.steps_slider.value)),
        )
        
        self.guidance_value_text = Text("", size=11)
        self.guidance_slider = Slider(
            min=0, max=20, divisions=40, value=7,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.guidance_value_text, "CFG {:.1f}", self.guidance_slider.value),
        )
        
        self.width_dropdown = Dropdown(
            label="Width",
            expand=True,
            options=[dropdown.Option(str(s), str(s)) for s in [512, 576, 640, 704, 768, 832, 896, 960, 1024]],
            value="512",
            on_change=self._on_dimension_change,
        )
        
        self.height_dropdown = Dropdown(
            label="Height",
            expand=True,
            options=[dropdown.Option(str(s), str(s)) for s in [512, 576, 640, 704, 768, 832, 896, 960, 1024]],
            value="768",
            on_change=self._on_dimension_change,
        )
        
        self.seed_field = TextField(
            label="Seed (-1 = random)",
            value="-1",
            width=160,
            height=50,
        )
        
        # Prompt fields
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
        
        # Buttons
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
        
        # Progress
        self.progress_bar = ProgressBar(value=0, width=300)
        self.progress_text = Text("", size=12)
        
        # Output
        self.result_image = Image(
            src="",
            width=512,
            height=512,
            fit=ft.ImageFit.CONTAIN,
            visible=False,
        )
        self.result_info = Text("", size=11, color=Colors.GREY_400)
        
        # No model banner
        self.no_model_banner = Container(
            content=Row([
                ft.Icon(Icons.WARNING_AMBER_ROUNDED, size=24, color=Colors.AMBER_400),
                Text("🎨 No image model loaded! Load a model from the header dropdown to generate images.", 
                     size=14, color=Colors.AMBER_300),
            ], spacing=10, alignment=MainAxisAlignment.CENTER),
            bgcolor=Colors.with_opacity(0.3, Colors.AMBER_900),
            padding=padding.all(12),
            border_radius=border_radius.all(8),
            visible=True,
        )
        
        # Main container
        self.main_container = Container(expand=True)
        
        # Initialize labels
        self._update_slider_label(self.lora_strength_value_text, "LoRA {:.2f}", self.lora_strength.value)
        self._update_slider_label(self.steps_value_text, "{} steps", int(self.steps_slider.value))
        self._update_slider_label(self.guidance_value_text, "CFG {:.1f}", self.guidance_slider.value)
    
    def _on_prompt_change(self, e):
        """Handle prompt field change."""
        self._update_save_prompt_btn()
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _update_save_prompt_btn(self):
        """Update save prompt button enabled state."""
        has_prompt = self.prompt_field.value and self.prompt_field.value.strip()
        self.save_prompt_btn.disabled = not has_prompt
        try:
            self.page.update()
        except:
            pass
    
    def _generate(self, e):
        """Generate image."""
        prompt = self.prompt_field.value
        
        if not prompt or not prompt.strip():
            self.page.snack_bar = SnackBar(content=Text("Please enter a prompt"))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        current_model, current_key = get_current_model()
        if current_model is None:
            self.page.snack_bar = SnackBar(
                content=Text("❌ No model loaded! Load a model first."),
                bgcolor=Colors.RED_900,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        app_state.is_generating = True
        app_state.cancel_requested = False
        
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
                    cancel_callback=lambda: app_state.cancel_requested,
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
                    self._rebuild_layout()
                
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
        """Save current prompt to library."""
        prompt = self.prompt_field.value.strip()
        if not prompt:
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
    
    def _build_settings_panel(self) -> Container:
        """Build the compact settings panel."""
        return Container(
            content=Column([
                Row([self.layout_dropdown], alignment=MainAxisAlignment.START),
                Container(height=10),
                Text("🎭 LoRA", weight=FontWeight.BOLD, size=13),
                self.lora_dropdown,
                Text("Strength:", size=11),
                self.lora_strength,
                self.lora_strength_value_text,
                Container(height=8),
                Text("⚙️ Settings", weight=FontWeight.BOLD, size=13),
                Text("Steps:", size=11),
                self.steps_slider,
                self.steps_value_text,
                Container(height=8),
                Text("CFG:", size=11),
                self.guidance_slider,
                self.guidance_value_text,
                Container(height=8),
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
            Container(expand=True),
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
        img_width = int(self.width_dropdown.value)
        img_height = int(self.height_dropdown.value)
        
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
        """Build layout with image on the right side."""
        return Row([
            self._build_settings_panel(),
            Container(
                content=self._build_prompt_section(),
                width=350,
                padding=padding.only(left=15, right=15),
            ),
            Container(
                content=self._build_image_output(max_height=600),
                expand=True,
                alignment=ft.alignment.center,
            ),
        ], expand=True, spacing=0, vertical_alignment=CrossAxisAlignment.STRETCH)
    
    def _build_bottom_layout(self) -> Column:
        """Build layout with image on the bottom."""
        return Column([
            Row([
                self._build_settings_panel(),
                Container(
                    content=self._build_prompt_section(),
                    expand=True,
                    padding=padding.only(left=15),
                ),
            ]),
            Container(
                content=self._build_image_output(max_width=800),
                expand=True,
                alignment=ft.alignment.center,
                padding=padding.only(top=15),
            ),
        ], expand=True, spacing=0)
    
    def build(self) -> Container:
        """Build the tab layout."""
        layout = self._get_optimal_layout()
        if layout == self.LAYOUT_RIGHT:
            self.main_container.content = self._build_side_layout()
        else:
            self.main_container.content = self._build_bottom_layout()
        
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
