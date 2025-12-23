"""
LoRA Manager Tab for KVGenius Desktop App.
Browse, import, delete, and edit LoRA models.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, TextField, ElevatedButton, 
    IconButton, Colors, Icons, ListView, Card, Dropdown, dropdown,
    MainAxisAlignment, CrossAxisAlignment, ScrollMode, SnackBar,
    FontWeight, padding, border_radius, AlertDialog, TextButton,
    Tabs, Tab, ProgressBar, FilePicker, FilePickerResultEvent,
    Divider, Image,
)

from ui.state import app_state

logger = logging.getLogger(__name__)

# LoRA directory
LORA_DIR = Path("./data/lora_models")
LORA_DIR.mkdir(parents=True, exist_ok=True)


class LoRAManagerTab:
    """LoRA Manager - browse, import, and manage LoRA models."""
    
    def __init__(self, page: Page):
        self.page = page
        self.selected_lora: Optional[str] = None  # Currently selected LoRA name
        self.filter_use_case: str = "all"  # all, image, text
        self.filter_base_model: str = "all"  # all, sd15, sdxl, mistral, llama
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        # LoRA list
        self.lora_list = ListView(spacing=8, padding=padding.all(10), expand=True)
        
        # Use case filter (Image vs Text)
        self.use_case_dropdown = Dropdown(
            label="LoRA Type",
            width=140,
            options=[
                dropdown.Option("all", "All Types"),
                dropdown.Option("image", "🎨 Image"),
                dropdown.Option("text", "💬 Text/Chat"),
            ],
            value="all",
            on_change=self._on_filter_change,
        )
        
        # Base model filter
        self.filter_dropdown = Dropdown(
            label="Base Model",
            width=140,
            options=[
                dropdown.Option("all", "All Models"),
                dropdown.Option("sd15", "SD 1.5"),
                dropdown.Option("sdxl", "SDXL"),
                dropdown.Option("mistral", "Mistral"),
                dropdown.Option("llama", "Llama"),
                dropdown.Option("unknown", "Unknown"),
            ],
            value="all",
            on_change=self._on_filter_change,
        )
        
        # Search field
        self.search_field = TextField(
            label="🔍 Search LoRAs",
            width=250,
            on_change=self._on_search_change,
        )
        
        # Refresh button
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh LoRA list",
            on_click=lambda e: self._refresh(),
        )
        
        # Import button
        self.import_btn = ElevatedButton(
            "📥 Import LoRA",
            icon=Icons.FILE_UPLOAD,
            on_click=self._open_import_dialog,
        )
        
        # Status text
        self.status_text = Text("", size=11, color=Colors.GREY_400)
        
        # File picker for import
        self.file_picker = FilePicker(
            on_result=self._on_file_picked,
        )
        
        # === DETAIL PANEL (right side) ===
        self.detail_name = Text("Select a LoRA", size=18, weight=FontWeight.BOLD)
        self.detail_path = Text("", size=11, color=Colors.GREY_500, selectable=True)
        self.detail_use_case = Text("", size=12)
        self.detail_base_model = Text("", size=12)
        self.detail_triggers = Text("", size=12, selectable=True)
        self.detail_rank = Text("", size=12)
        self.detail_description = Text("", size=12, color=Colors.GREY_400)
        self.detail_size = Text("", size=11, color=Colors.GREY_500)
        
        # Preview image (if available)
        self.preview_image = Image(
            src="",
            width=200,
            height=200,
            fit=ft.ImageFit.CONTAIN,
            visible=False,
        )
        
        # Edit fields (shown when editing)
        self.edit_name = TextField(label="Name", width=300)
        self.edit_use_case = Dropdown(
            label="LoRA Type",
            width=200,
            options=[
                dropdown.Option("image", "🎨 Image (SD/SDXL)"),
                dropdown.Option("text", "💬 Text/Chat (LLM)"),
            ],
            value="image",
            on_change=self._on_edit_use_case_change,
        )
        self.edit_triggers = TextField(label="Trigger Words (comma separated)", width=300)
        self.edit_base_model = Dropdown(
            label="Base Model",
            width=200,
            options=[
                dropdown.Option("sd15", "SD 1.5"),
                dropdown.Option("sdxl", "SDXL"),
                dropdown.Option("unknown", "Unknown"),
            ],
            value="unknown",
        )
        self.edit_description = TextField(
            label="Description",
            multiline=True,
            min_lines=2,
            max_lines=4,
            width=300,
        )
        
        # Action buttons
        self.save_btn = ElevatedButton(
            "💾 Save Changes",
            icon=Icons.SAVE,
            on_click=self._save_metadata,
            visible=False,
        )
        self.delete_btn = ElevatedButton(
            "🗑️ Delete LoRA",
            icon=Icons.DELETE,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._confirm_delete,
            visible=False,
        )
        self.open_folder_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=self._open_lora_folder,
            visible=False,
        )
        
        # Edit mode toggle
        self.edit_mode = False
        self.edit_toggle_btn = ElevatedButton(
            "✏️ Edit",
            icon=Icons.EDIT,
            on_click=self._toggle_edit_mode,
            visible=False,
        )
        
        # Detail view container (switches between view and edit)
        self.detail_view = Column([], spacing=8, visible=True)
        self.edit_view = Column([], spacing=8, visible=False)
    
    def _on_filter_change(self, e):
        """Handle filter dropdown change."""
        self.filter_use_case = self.use_case_dropdown.value
        self.filter_base_model = self.filter_dropdown.value
        self._refresh()
    
    def _on_edit_use_case_change(self, e):
        """Update base model options when use case changes."""
        if self.edit_use_case.value == "image":
            self.edit_base_model.options = [
                dropdown.Option("sd15", "SD 1.5"),
                dropdown.Option("sdxl", "SDXL"),
                dropdown.Option("unknown", "Unknown"),
            ]
            self.edit_triggers.visible = True
        else:
            self.edit_base_model.options = [
                dropdown.Option("mistral", "Mistral"),
                dropdown.Option("llama", "Llama"),
                dropdown.Option("llama3", "Llama 3"),
                dropdown.Option("unknown", "Unknown"),
            ]
            self.edit_triggers.visible = False  # Text LoRAs don't use trigger words
        self.edit_base_model.value = "unknown"
        self.page.update()
    
    def _on_search_change(self, e):
        """Handle search text change."""
        self._refresh()
    
    def _refresh(self):
        """Refresh the LoRA list."""
        self._load_loras()
        self.page.update()
    
    def _get_lora_info(self, lora_path: Path) -> Dict[str, Any]:
        """Get info for a LoRA folder."""
        info = {
            "name": lora_path.name,
            "path": str(lora_path),
            "use_case": "image",  # Default to image LoRA
            "base_model": "unknown",
            "trigger_words": [],
            "description": "",
            "rank": 0,
            "size_mb": 0.0,
            "preview_image": None,
        }
        
        # Check for adapter file
        adapter_file = lora_path / "adapter_model.safetensors"
        if adapter_file.exists():
            info["size_mb"] = adapter_file.stat().st_size / (1024 * 1024)
        else:
            # Check for any safetensors file
            safetensors = list(lora_path.glob("*.safetensors"))
            if safetensors:
                info["size_mb"] = sum(f.stat().st_size for f in safetensors) / (1024 * 1024)
        
        # Read metadata
        metadata_file = lora_path / "lora_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info["use_case"] = metadata.get("use_case", "image")
                info["base_model"] = metadata.get("base_model", "unknown")
                info["trigger_words"] = metadata.get("trigger_words", metadata.get("trigger_word", []))
                if isinstance(info["trigger_words"], str):
                    info["trigger_words"] = [info["trigger_words"]]
                info["description"] = metadata.get("description", "")
                info["rank"] = metadata.get("rank", 0)
            except Exception as e:
                logger.warning(f"Error reading metadata for {lora_path.name}: {e}")
        
        # Check for preview image
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            preview_file = lora_path / f"preview{ext}"
            if preview_file.exists():
                info["preview_image"] = str(preview_file)
                break
            # Also check for thumbnail
            thumb_file = lora_path / f"thumbnail{ext}"
            if thumb_file.exists():
                info["preview_image"] = str(thumb_file)
                break
        
        # Infer use_case and base model from name/path if unknown
        if info["use_case"] == "image" and info["base_model"] == "unknown":
            name_lower = lora_path.name.lower()
            # Check if it's a text/chat LoRA
            if "mistral" in name_lower or "llama" in name_lower or "chat" in name_lower:
                info["use_case"] = "text"
                if "mistral" in name_lower:
                    info["base_model"] = "mistral"
                elif "llama" in name_lower:
                    info["base_model"] = "llama"
            else:
                # Image LoRA - infer SD version
                if "xl" in name_lower or "sdxl" in name_lower:
                    info["base_model"] = "sdxl"
                elif "sd15" in name_lower or "1.5" in name_lower:
                    info["base_model"] = "sd15"
        
        return info
    
    def _load_loras(self):
        """Load all LoRAs from the directory."""
        self.lora_list.controls.clear()
        
        search_text = self.search_field.value.lower().strip() if self.search_field.value else ""
        
        lora_count = 0
        if LORA_DIR.exists():
            for item in sorted(LORA_DIR.iterdir()):
                if not item.is_dir():
                    continue
                
                # Check for adapter files
                has_adapter = (item / "adapter_model.safetensors").exists() or \
                             (item / "adapter_model.bin").exists() or \
                             list(item.glob("*.safetensors"))
                
                if not has_adapter:
                    continue
                
                info = self._get_lora_info(item)
                
                # Apply use case filter
                if self.filter_use_case != "all" and info["use_case"] != self.filter_use_case:
                    continue
                
                # Apply base model filter
                if self.filter_base_model != "all" and info["base_model"] != self.filter_base_model:
                    continue
                
                # Apply search
                if search_text:
                    if search_text not in info["name"].lower() and \
                       search_text not in info["description"].lower() and \
                       not any(search_text in tw.lower() for tw in info["trigger_words"]):
                        continue
                
                lora_count += 1
                self._add_lora_card(info)
        
        if lora_count == 0:
            self.lora_list.controls.append(
                Container(
                    content=Text(
                        "No LoRAs found. Import one using the button above!",
                        color=Colors.GREY_500,
                        italic=True,
                    ),
                    padding=padding.all(20),
                )
            )
        
        self.status_text.value = f"Found {lora_count} LoRA(s)"
    
    def _add_lora_card(self, info: Dict[str, Any]):
        """Add a LoRA card to the list."""
        name = info["name"]
        use_case = info["use_case"]
        base_model = info["base_model"].upper()
        triggers = ", ".join(info["trigger_words"][:3]) if info["trigger_words"] else "No triggers"
        size = f"{info['size_mb']:.1f} MB"
        
        # Use case badge
        if use_case == "image":
            use_case_text = "🎨 Image"
            use_case_color = Colors.GREEN_700
        else:
            use_case_text = "💬 Text"
            use_case_color = Colors.ORANGE_700
        
        # Color code by base model
        if info["base_model"] == "sdxl":
            badge_color = Colors.PURPLE_700
        elif info["base_model"] == "sd15":
            badge_color = Colors.BLUE_700
        elif info["base_model"] in ["mistral", "llama"]:
            badge_color = Colors.TEAL_700
        else:
            badge_color = Colors.GREY_700
        
        is_selected = self.selected_lora == name
        
        card = Card(
            content=Container(
                content=Row([
                    Column([
                        Text(name, weight=FontWeight.BOLD, size=14),
                        Row([
                            Container(
                                content=Text(use_case_text, size=10),
                                bgcolor=use_case_color,
                                padding=padding.symmetric(horizontal=8, vertical=3),
                                border_radius=border_radius.all(5),
                            ),
                            Container(
                                content=Text(base_model, size=10),
                                bgcolor=badge_color,
                                padding=padding.symmetric(horizontal=8, vertical=3),
                                border_radius=border_radius.all(5),
                            ),
                            Text(size, size=10, color=Colors.GREY_500),
                        ], spacing=8),
                        Text(
                            f"🎯 {triggers}" if triggers != "No triggers" else triggers,
                            size=11,
                            color=Colors.GREY_400,
                            max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ) if use_case == "image" else Container(),
                    ], expand=True, spacing=4),
                    IconButton(
                        icon=Icons.CHEVRON_RIGHT,
                        on_click=lambda e, n=name: self._select_lora(n),
                    ),
                ], spacing=10),
                padding=padding.all(12),
                bgcolor=Colors.with_opacity(0.2, Colors.PRIMARY) if is_selected else None,
                border_radius=border_radius.all(8),
                on_click=lambda e, n=name: self._select_lora(n),
                ink=True,
            ),
        )
        self.lora_list.controls.append(card)
    
    def _select_lora(self, name: str):
        """Select a LoRA and show its details."""
        self.selected_lora = name
        self.edit_mode = False
        self._update_detail_panel()
        self._refresh()  # Refresh to update selection highlight
    
    def _update_detail_panel(self):
        """Update the detail panel with selected LoRA info."""
        if not self.selected_lora:
            self.detail_name.value = "Select a LoRA"
            self.detail_path.value = ""
            self.detail_use_case.value = ""
            self.detail_base_model.value = ""
            self.detail_triggers.value = ""
            self.detail_rank.value = ""
            self.detail_description.value = ""
            self.detail_size.value = ""
            self.preview_image.visible = False
            self.save_btn.visible = False
            self.delete_btn.visible = False
            self.edit_toggle_btn.visible = False
            self.open_folder_btn.visible = False
            self.detail_view.visible = True
            self.edit_view.visible = False
            return
        
        lora_path = LORA_DIR / self.selected_lora
        info = self._get_lora_info(lora_path)
        
        # Update view mode
        self.detail_name.value = info["name"]
        self.detail_path.value = f"📁 {info['path']}"
        use_case_display = "🎨 Image Generation" if info["use_case"] == "image" else "💬 Text/Chat"
        self.detail_use_case.value = f"🏷️ Type: {use_case_display}"
        self.detail_base_model.value = f"🏗️ Base Model: {info['base_model'].upper()}"
        triggers_str = ", ".join(info["trigger_words"]) if info["trigger_words"] else "None"
        self.detail_triggers.value = f"🎯 Trigger Words: {triggers_str}" if info["use_case"] == "image" else ""
        self.detail_rank.value = f"📊 Rank: {info['rank']}" if info["rank"] else ""
        self.detail_description.value = info["description"] if info["description"] else "No description"
        self.detail_size.value = f"💾 Size: {info['size_mb']:.1f} MB"
        
        # Preview image
        if info["preview_image"] and os.path.exists(info["preview_image"]):
            self.preview_image.src = info["preview_image"]
            self.preview_image.visible = True
        else:
            self.preview_image.visible = False
        
        # Update edit mode fields
        self.edit_name.value = info["name"]
        self.edit_use_case.value = info["use_case"]
        self._on_edit_use_case_change(None)  # Update base model options
        self.edit_base_model.value = info["base_model"]
        self.edit_triggers.value = ", ".join(info["trigger_words"])
        self.edit_description.value = info["description"]
        
        # Show buttons
        self.save_btn.visible = self.edit_mode
        self.delete_btn.visible = True
        self.edit_toggle_btn.visible = True
        self.open_folder_btn.visible = True
        
        # Toggle view/edit
        self.detail_view.visible = not self.edit_mode
        self.edit_view.visible = self.edit_mode
        self.edit_toggle_btn.text = "❌ Cancel" if self.edit_mode else "✏️ Edit"
    
    def _toggle_edit_mode(self, e):
        """Toggle between view and edit mode."""
        self.edit_mode = not self.edit_mode
        self._update_detail_panel()
        self.page.update()
    
    def _save_metadata(self, e):
        """Save edited metadata."""
        if not self.selected_lora:
            return
        
        lora_path = LORA_DIR / self.selected_lora
        metadata_file = lora_path / "lora_metadata.json"
        
        # Load existing metadata
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Update fields
        new_name = self.edit_name.value.strip()
        metadata["use_case"] = self.edit_use_case.value
        metadata["base_model"] = self.edit_base_model.value
        metadata["description"] = self.edit_description.value.strip()
        
        # Only save triggers for image LoRAs
        if self.edit_use_case.value == "image":
            metadata["trigger_words"] = [tw.strip() for tw in self.edit_triggers.value.split(",") if tw.strip()]
            metadata["trigger_word"] = metadata["trigger_words"][0] if metadata["trigger_words"] else ""
        else:
            metadata["trigger_words"] = []
            metadata["trigger_word"] = ""
        
        # Save metadata
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Rename folder if name changed
            if new_name != self.selected_lora:
                new_path = LORA_DIR / new_name
                if not new_path.exists():
                    lora_path.rename(new_path)
                    self.selected_lora = new_name
            
            self.page.snack_bar = SnackBar(content=Text(f"✅ Saved: {new_name}"))
            self.page.snack_bar.open = True
            self.edit_mode = False
            self._refresh()
            self._update_detail_panel()
            self.page.update()
            
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Error saving: {str(ex)}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def _confirm_delete(self, e):
        """Show delete confirmation dialog."""
        if not self.selected_lora:
            return
        
        def do_delete(e):
            self._delete_lora()
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete LoRA?"),
            content=Text(f"Are you sure you want to delete '{self.selected_lora}'?\n\nThis will permanently remove all files."),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Delete", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _delete_lora(self):
        """Delete the selected LoRA."""
        if not self.selected_lora:
            return
        
        lora_path = LORA_DIR / self.selected_lora
        try:
            shutil.rmtree(lora_path)
            self.page.snack_bar = SnackBar(content=Text(f"🗑️ Deleted: {self.selected_lora}"))
            self.page.snack_bar.open = True
            self.selected_lora = None
            self._refresh()
            self._update_detail_panel()
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Error deleting: {str(ex)}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
        
        self.page.update()
    
    def _open_lora_folder(self, e):
        """Open the LoRA folder in file explorer."""
        if not self.selected_lora:
            return
        
        lora_path = LORA_DIR / self.selected_lora
        import subprocess
        import sys
        
        if sys.platform == "win32":
            subprocess.run(["explorer", str(lora_path)])
        elif sys.platform == "darwin":
            subprocess.run(["open", str(lora_path)])
        else:
            subprocess.run(["xdg-open", str(lora_path)])
    
    def _open_import_dialog(self, e):
        """Open the file picker for importing a LoRA."""
        self.page.overlay.append(self.file_picker)
        self.file_picker.pick_files(
            allowed_extensions=["safetensors"],
            dialog_title="Select LoRA file (.safetensors)",
        )
        self.page.update()
    
    def _on_file_picked(self, e: FilePickerResultEvent):
        """Handle file picker result."""
        if not e.files:
            return
        
        file_path = e.files[0].path
        file_name = Path(file_path).stem
        
        # Show import dialog
        self._show_import_config_dialog(file_path, file_name)
    
    def _show_import_config_dialog(self, file_path: str, default_name: str):
        """Show dialog to configure imported LoRA."""
        name_field = TextField(label="LoRA Name", value=default_name, width=300)
        
        use_case_dropdown = Dropdown(
            label="LoRA Type",
            width=200,
            options=[
                dropdown.Option("image", "🎨 Image (SD/SDXL)"),
                dropdown.Option("text", "💬 Text/Chat (LLM)"),
            ],
            value="image",
        )
        
        trigger_field = TextField(label="Trigger Word(s)", hint_text="e.g., style_name", width=300)
        
        base_model_dropdown = Dropdown(
            label="Base Model",
            width=200,
            options=[
                dropdown.Option("auto", "Auto-detect"),
                dropdown.Option("sd15", "SD 1.5"),
                dropdown.Option("sdxl", "SDXL"),
            ],
            value="auto",
        )
        
        def on_use_case_change(e):
            if use_case_dropdown.value == "image":
                base_model_dropdown.options = [
                    dropdown.Option("auto", "Auto-detect"),
                    dropdown.Option("sd15", "SD 1.5"),
                    dropdown.Option("sdxl", "SDXL"),
                ]
                trigger_field.visible = True
            else:
                base_model_dropdown.options = [
                    dropdown.Option("auto", "Auto-detect"),
                    dropdown.Option("mistral", "Mistral"),
                    dropdown.Option("llama", "Llama"),
                    dropdown.Option("llama3", "Llama 3"),
                ]
                trigger_field.visible = False
            base_model_dropdown.value = "auto"
            self.page.update()
        
        use_case_dropdown.on_change = on_use_case_change
        
        desc_field = TextField(label="Description (optional)", width=300)
        
        def do_import(e):
            name = name_field.value.strip().replace(" ", "_")
            if not name:
                self.page.snack_bar = SnackBar(
                    content=Text("Please enter a name"),
                    bgcolor=Colors.ORANGE_700,
                )
                self.page.snack_bar.open = True
                self.page.update()
                return
            
            self._import_lora(
                file_path,
                name,
                use_case_dropdown.value,
                trigger_field.value.strip() if use_case_dropdown.value == "image" else "",
                base_model_dropdown.value,
                desc_field.value.strip(),
            )
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Import LoRA"),
            content=Container(
                content=Column([
                    name_field,
                    use_case_dropdown,
                    trigger_field,
                    base_model_dropdown,
                    desc_field,
                ], spacing=15),
                width=350,
            ),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton("Import", on_click=do_import),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _import_lora(self, file_path: str, name: str, use_case: str, trigger_word: str, base_model: str, description: str):
        """Import a LoRA file."""
        try:
            # Create LoRA directory
            lora_dir = LORA_DIR / name
            lora_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            dest_file = lora_dir / "adapter_model.safetensors"
            shutil.copy2(file_path, dest_file)
            
            # Detect rank from file
            rank = 16  # default
            try:
                from safetensors import safe_open
                with safe_open(str(dest_file), framework="pt") as f:
                    for key in f.keys():
                        if "lora_down" in key.lower() or "lora_a" in key.lower():
                            tensor = f.get_tensor(key)
                            rank = min(tensor.shape)
                            break
            except Exception as ex:
                logger.warning(f"Could not detect LoRA rank: {ex}")
            
            # Auto-detect base model
            detected_base = base_model
            if base_model == "auto":
                file_size = os.path.getsize(dest_file) / (1024 * 1024)
                fname_lower = os.path.basename(file_path).lower()
                
                if use_case == "image":
                    if "xl" in fname_lower or "sdxl" in fname_lower or file_size > 100:
                        detected_base = "sdxl"
                    else:
                        detected_base = "sd15"
                else:
                    # Text LoRA - try to detect from filename
                    if "mistral" in fname_lower:
                        detected_base = "mistral"
                    elif "llama3" in fname_lower:
                        detected_base = "llama3"
                    elif "llama" in fname_lower:
                        detected_base = "llama"
                    else:
                        detected_base = "unknown"
            
            # Create metadata
            metadata = {
                "name": name,
                "use_case": use_case,
                "base_model": detected_base,
                "rank": rank,
                "description": description,
                "source": "imported",
                "imported_from": os.path.basename(file_path),
            }
            
            # Only add trigger words for image LoRAs
            if use_case == "image":
                metadata["trigger_word"] = trigger_word
                metadata["trigger_words"] = [tw.strip() for tw in trigger_word.split(",") if tw.strip()] if trigger_word else []
            
            metadata_file = lora_dir / "lora_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Create adapter_config.json for compatibility (image LoRAs only)
            if use_case == "image":
                adapter_config = {
                    "peft_type": "LORA",
                    "base_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0" if detected_base == "sdxl" else "runwayml/stable-diffusion-v1-5",
                    "r": rank,
                    "lora_alpha": rank * 2,
                    "lora_dropout": 0.0,
                    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                }
                
                config_file = lora_dir / "adapter_config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(adapter_config, f, indent=2)
            
            self.page.snack_bar = SnackBar(content=Text(f"✅ Imported: {name}"))
            self.page.snack_bar.open = True
            
            self.selected_lora = name
            self._refresh()
            self._update_detail_panel()
            
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Import failed: {str(ex)}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
        
        self.page.update()
    
    def build(self) -> Container:
        """Build the LoRA Manager tab."""
        self._load_loras()
        self._update_detail_panel()
        
        # Build detail view content
        self.detail_view.controls = [
            self.detail_name,
            self.detail_path,
            Divider(),
            self.detail_use_case,
            self.detail_base_model,
            self.detail_triggers,
            self.detail_rank,
            self.detail_size,
            Divider(),
            Text("Description:", size=12, weight=FontWeight.BOLD),
            self.detail_description,
            self.preview_image,
        ]
        
        # Build edit view content
        self.edit_view.controls = [
            Text("Edit LoRA Metadata", size=16, weight=FontWeight.BOLD),
            self.edit_name,
            self.edit_use_case,
            self.edit_triggers,
            self.edit_base_model,
            self.edit_description,
            Row([
                self.save_btn,
            ], spacing=10),
        ]
        
        # Header
        header = Row([
            Text("🎨 LoRA Manager", size=18, weight=FontWeight.BOLD),
            Container(expand=True),
            self.search_field,
            self.use_case_dropdown,
            self.filter_dropdown,
            self.import_btn,
            self.refresh_btn,
        ], alignment=MainAxisAlignment.SPACE_BETWEEN)
        
        # Main layout - list on left, details on right
        return Container(
            content=Column([
                header,
                Row([
                    self.status_text,
                ]),
                Row([
                    # Left: LoRA list
                    Container(
                        content=self.lora_list,
                        expand=2,
                        bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
                        border_radius=border_radius.all(10),
                    ),
                    # Right: Detail panel
                    Container(
                        content=Column([
                            Row([
                                self.edit_toggle_btn,
                                self.open_folder_btn,
                                Container(expand=True),
                                self.delete_btn,
                            ]),
                            self.detail_view,
                            self.edit_view,
                        ], spacing=10, scroll=ScrollMode.AUTO),
                        expand=1,
                        padding=padding.all(15),
                        bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
                        border_radius=border_radius.all(10),
                    ),
                ], expand=True, spacing=15),
            ], spacing=10),
            padding=padding.all(15),
            expand=True,
        )
