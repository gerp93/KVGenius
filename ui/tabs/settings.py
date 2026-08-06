"""
Settings Tab - application settings and preferences
"""
from typing import Optional
from pathlib import Path
import os
import logging
import threading

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, TextField, ElevatedButton,
    IconButton, Colors, Icons, ScrollMode, FontWeight, padding,
    Slider, Dropdown, dropdown, SnackBar, AlertDialog, TextButton,
)

from updater import CURRENT_VERSION, check_for_update, check_and_apply_update

logger = logging.getLogger(__name__)


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
        from core.config import get_setting, set_setting
        
        self.CACHE_DIR = CACHE_DIR
        self.GENERATED_IMAGES_DIR = GENERATED_IMAGES_DIR
        self.LORA_DIR = LORA_DIR
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        self.get_setting = get_setting
        self.set_setting = set_setting
        
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

        # Update check
        self.version_text = Text(f"Version {CURRENT_VERSION}", size=12, color=Colors.GREY_400)
        self.check_updates_btn = ElevatedButton(
            "Check for Updates",
            icon=Icons.SYSTEM_UPDATE,
            on_click=self._check_for_updates,
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
        self.set_setting("default_steps", int(self.default_steps.value))
        self.set_setting("default_guidance", self.default_guidance.value)
        self.set_setting("default_width", int(self.default_width.value))
        self.set_setting("default_height", int(self.default_height.value))
        
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
    
    def _check_for_updates(self, e):
        """Check GitHub Releases for a newer build (kvg_updater bundle mode)."""
        self.check_updates_btn.disabled = True
        self.page.update()
        threading.Thread(target=self._check_for_updates_worker, args=(True,), daemon=True).start()

    def check_for_updates_silently(self):
        """Startup check: prompts only if an update is actually available."""
        threading.Thread(target=self._check_for_updates_worker, args=(False,), daemon=True).start()

    def _check_for_updates_worker(self, manual: bool):
        try:
            update = check_for_update()
        except Exception as e:
            update = None
            logger.error(f"Update check failed: {e}")

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
        except Exception as e:
            logger.error(f"Update failed: {e}")
            self.page.snack_bar = SnackBar(content=Text(f"Update failed: {e}"), bgcolor=Colors.RED_700)
            self.page.snack_bar.open = True
            self.page.update()

    def _clear_cache(self):
        """Clear the model cache directory."""
        import shutil
        
        try:
            if self.CACHE_DIR.exists():
                for item in self.CACHE_DIR.iterdir():
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
        total = 0
        if self.CACHE_DIR.exists():
            for root, dirs, files in os.walk(self.CACHE_DIR):
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
