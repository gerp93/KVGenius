"""
Dataset Manager Tab for KVGenius Flet App
Manages training datasets with images and captions for LoRA training.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Callable
from datetime import datetime

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField, 
    ElevatedButton, IconButton, TextButton, Card, Image,
    Icons, Colors, FontWeight, ScrollMode, AlertDialog, 
    SnackBar, ListView, GridView, Dropdown, dropdown,
    ProgressBar, Divider, padding, border_radius, alignment,
    FilePicker, FilePickerResultEvent, FilePickerFileType,
    ControlState, ButtonStyle, MainAxisAlignment, CrossAxisAlignment,
)

logger = logging.getLogger(__name__)

# Directories
DATASETS_DIR = Path("./data/training_datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


class DatasetManagerTab:
    """Dataset Manager - create and manage training datasets with images and captions."""
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    GRID_COLUMNS = 4
    
    def __init__(self, page: Page):
        self.page = page
        self.datasets: List[Dict] = []
        self.selected_dataset: Optional[str] = None
        self.selected_image: Optional[Dict] = None
        self.images: List[Dict] = []
        self.editing_caption = False
        self._build_ui()
    
    def _build_ui(self):
        """Build all UI components."""
        # File picker for importing images
        self.file_picker = FilePicker(on_result=self._on_files_picked)
        
        # Dataset list panel (left)
        self.dataset_list = ListView(spacing=8, padding=10, expand=True)
        
        # Search/filter
        self.search_field = TextField(
            label="🔍 Search",
            width=200,
            on_change=self._on_search_change,
        )
        
        # Action buttons
        self.new_dataset_btn = ElevatedButton(
            "➕ New Dataset",
            icon=Icons.CREATE_NEW_FOLDER,
            on_click=self._show_create_dialog,
        )
        
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh datasets",
            on_click=lambda e: self._refresh(),
        )
        
        # Image grid (center)
        self.image_grid = GridView(
            runs_count=self.GRID_COLUMNS,
            max_extent=180,
            child_aspect_ratio=0.85,
            spacing=10,
            run_spacing=10,
            expand=True,
            padding=padding.all(10),
        )
        
        # Import images button
        self.import_btn = ElevatedButton(
            "📥 Add Images",
            icon=Icons.ADD_PHOTO_ALTERNATE,
            on_click=self._pick_images,
            disabled=True,
        )
        
        # Delete dataset button
        self.delete_dataset_btn = IconButton(
            icon=Icons.DELETE_FOREVER,
            tooltip="Delete dataset",
            icon_color=Colors.RED_400,
            on_click=self._confirm_delete_dataset,
            disabled=True,
        )
        
        # Open folder button
        self.open_folder_btn = IconButton(
            icon=Icons.FOLDER_OPEN,
            tooltip="Open folder",
            on_click=self._open_dataset_folder,
            disabled=True,
        )
        
        # Status text
        self.status_text = Text("", size=11, color=Colors.GREY_400)
        self.dataset_info_text = Text("", size=12)
        
        # Image detail panel (right)
        self.detail_image = Image(
            src="",
            width=250,
            height=250,
            fit=ft.ImageFit.CONTAIN,
            border_radius=border_radius.all(8),
        )
        
        self.detail_filename = Text("", size=12, weight=FontWeight.BOLD)
        self.detail_size = Text("", size=11, color=Colors.GREY_400)
        
        # Caption editor
        self.caption_field = TextField(
            label="Caption / Prompt",
            multiline=True,
            min_lines=4,
            max_lines=8,
            expand=True,
            on_change=self._on_caption_change,
        )
        
        self.save_caption_btn = ElevatedButton(
            "💾 Save Caption",
            icon=Icons.SAVE,
            on_click=self._save_caption,
            disabled=True,
        )
        
        self.delete_image_btn = ElevatedButton(
            "🗑️ Delete Image",
            icon=Icons.DELETE,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._confirm_delete_image,
            disabled=True,
        )
        
        # Detail panel container
        self.detail_panel = Container(
            content=Column([
                Text("Select an image", size=14, color=Colors.GREY_500),
            ], alignment=MainAxisAlignment.CENTER, horizontal_alignment=CrossAxisAlignment.CENTER),
            expand=True,
            padding=padding.all(15),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(10),
        )
        
        # No dataset selected message
        self.no_dataset_msg = Container(
            content=Column([
                ft.Icon(Icons.FOLDER_OFF, size=48, color=Colors.GREY_500),
                Text("Select a dataset to view images", size=14, color=Colors.GREY_500),
            ], alignment=MainAxisAlignment.CENTER, horizontal_alignment=CrossAxisAlignment.CENTER, spacing=10),
            expand=True,
            alignment=alignment.center,
        )
        
        # Empty dataset message
        self.empty_dataset_msg = Container(
            content=Column([
                ft.Icon(Icons.IMAGE_NOT_SUPPORTED, size=48, color=Colors.GREY_500),
                Text("No images in this dataset", size=14, color=Colors.GREY_500),
                Text("Click 'Add Images' to import training images", size=12, color=Colors.GREY_600),
            ], alignment=MainAxisAlignment.CENTER, horizontal_alignment=CrossAxisAlignment.CENTER, spacing=10),
            expand=True,
            alignment=alignment.center,
        )
    
    def _refresh(self):
        """Refresh the dataset list."""
        self._load_datasets()
        if self.selected_dataset:
            self._load_images()
        if self.page:
            self.page.update()
    
    def _on_search_change(self, e):
        """Filter datasets based on search."""
        self._load_datasets()
        self.page.update()
    
    def _load_datasets(self):
        """Load all datasets from disk."""
        self.datasets = []
        search_text = self.search_field.value.lower() if self.search_field.value else ""
        
        for item in DATASETS_DIR.iterdir():
            if item.is_dir():
                # Get dataset info
                info = self._get_dataset_info(item.name)
                if search_text and search_text not in item.name.lower():
                    continue
                
                # Count images
                images_dir = item / "images"
                image_count = 0
                if images_dir.exists():
                    image_count = sum(1 for f in images_dir.iterdir() 
                                     if f.suffix.lower() in self.SUPPORTED_EXTENSIONS)
                
                self.datasets.append({
                    "name": item.name,
                    "path": str(item),
                    "image_count": image_count,
                    "created_at": info.get("created_at", ""),
                })
        
        # Sort by name
        self.datasets.sort(key=lambda x: x["name"].lower())
        
        # Update status
        self.status_text.value = f"{len(self.datasets)} dataset(s)"
        
        # Rebuild list
        self._build_dataset_list()
    
    def _get_dataset_info(self, name: str) -> Dict:
        """Get dataset metadata from JSON file."""
        metadata_path = DATASETS_DIR / name / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_dataset_info(self, name: str, info: Dict):
        """Save dataset metadata to JSON file."""
        metadata_path = DATASETS_DIR / name / "metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
        except Exception as ex:
            logger.error(f"Error saving dataset info: {ex}")
    
    def _build_dataset_list(self):
        """Build the dataset list UI."""
        self.dataset_list.controls.clear()
        
        if not self.datasets:
            self.dataset_list.controls.append(
                Container(
                    content=Column([
                        ft.Icon(Icons.FOLDER_OFF, size=32, color=Colors.GREY_600),
                        Text("No datasets found", color=Colors.GREY_500),
                        Text("Create one to get started", size=11, color=Colors.GREY_600),
                    ], horizontal_alignment=CrossAxisAlignment.CENTER, spacing=5),
                    padding=padding.all(20),
                    alignment=alignment.center,
                )
            )
            return
        
        for ds in self.datasets:
            is_selected = ds["name"] == self.selected_dataset
            
            card = Container(
                content=Row([
                    ft.Icon(Icons.FOLDER, color=Colors.AMBER_400 if is_selected else Colors.GREY_500),
                    Column([
                        Text(ds["name"], weight=FontWeight.BOLD if is_selected else FontWeight.NORMAL, 
                             size=13),
                        Text(f"{ds['image_count']} images", size=10, color=Colors.GREY_400),
                    ], spacing=2, expand=True),
                ], spacing=10),
                padding=padding.all(10),
                bgcolor=Colors.BLUE_900 if is_selected else Colors.with_opacity(0.1, Colors.WHITE),
                border_radius=border_radius.all(8),
                on_click=lambda e, name=ds["name"]: self._select_dataset(name),
                ink=True,
            )
            self.dataset_list.controls.append(card)
    
    def _select_dataset(self, name: str):
        """Select a dataset and load its images."""
        self.selected_dataset = name
        self.selected_image = None
        
        # Update buttons
        self.import_btn.disabled = False
        self.delete_dataset_btn.disabled = False
        self.open_folder_btn.disabled = False
        
        # Update info
        ds = next((d for d in self.datasets if d["name"] == name), None)
        if ds:
            self.dataset_info_text.value = f"📁 {name} - {ds['image_count']} images"
        
        self._build_dataset_list()
        self._load_images()
        self._update_detail_panel()
        self.page.update()
    
    def _load_images(self):
        """Load images from the selected dataset."""
        self.images = []
        
        if not self.selected_dataset:
            return
        
        images_dir = DATASETS_DIR / self.selected_dataset / "images"
        if not images_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # Load caption
            caption_path = img_path.with_suffix(".txt")
            caption = ""
            if caption_path.exists():
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                except:
                    pass
            
            # Get file size
            file_size = img_path.stat().st_size / 1024  # KB
            
            self.images.append({
                "path": str(img_path),
                "filename": img_path.name,
                "caption": caption,
                "size_kb": file_size,
            })
        
        self._build_image_grid()
    
    def _build_image_grid(self):
        """Build the image grid UI."""
        self.image_grid.controls.clear()
        
        if not self.images:
            return
        
        for img in self.images:
            is_selected = (self.selected_image and 
                          self.selected_image.get("path") == img["path"])
            has_caption = bool(img["caption"])
            
            # Caption indicator
            caption_indicator = Container(
                content=ft.Icon(
                    Icons.TEXT_SNIPPET if has_caption else Icons.TEXT_SNIPPET_OUTLINED,
                    size=16,
                    color=Colors.GREEN_400 if has_caption else Colors.GREY_600,
                ),
                bgcolor=Colors.with_opacity(0.8, Colors.BLACK),
                padding=padding.all(4),
                border_radius=border_radius.all(4),
            )
            
            card = Container(
                content=Column([
                    Stack([
                        Container(
                            content=Image(
                                src=img["path"],
                                width=150,
                                height=120,
                                fit=ft.ImageFit.COVER,
                                border_radius=border_radius.all(6),
                            ),
                            bgcolor=Colors.GREY_900,
                            border_radius=border_radius.all(6),
                        ),
                        Container(
                            content=caption_indicator,
                            alignment=alignment.bottom_right,
                            padding=padding.all(5),
                        ),
                    ]),
                    Text(
                        img["filename"][:20] + "..." if len(img["filename"]) > 20 else img["filename"],
                        size=10,
                        color=Colors.GREY_300,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ], spacing=5, horizontal_alignment=CrossAxisAlignment.CENTER),
                padding=padding.all(8),
                bgcolor=Colors.BLUE_900 if is_selected else Colors.with_opacity(0.1, Colors.WHITE),
                border_radius=border_radius.all(8),
                border=ft.border.all(2, Colors.BLUE_400) if is_selected else None,
                on_click=lambda e, i=img: self._select_image(i),
                ink=True,
            )
            self.image_grid.controls.append(card)
    
    def _select_image(self, img: Dict):
        """Select an image to view/edit."""
        self.selected_image = img
        self._build_image_grid()
        self._update_detail_panel()
        self.page.update()
    
    def _update_detail_panel(self):
        """Update the image detail panel."""
        if not self.selected_image:
            self.detail_panel.content = Column([
                Text("Select an image to view details", size=14, color=Colors.GREY_500),
            ], alignment=MainAxisAlignment.CENTER, horizontal_alignment=CrossAxisAlignment.CENTER,
               expand=True)
            self.save_caption_btn.disabled = True
            self.delete_image_btn.disabled = True
            return
        
        img = self.selected_image
        
        # Update fields
        self.detail_image.src = img["path"]
        self.detail_filename.value = img["filename"]
        self.detail_size.value = f"📁 {img['size_kb']:.1f} KB"
        self.caption_field.value = img["caption"]
        
        # Enable buttons
        self.save_caption_btn.disabled = False
        self.delete_image_btn.disabled = False
        
        self.detail_panel.content = Column([
            self.detail_image,
            self.detail_filename,
            self.detail_size,
            Divider(),
            Text("Caption:", size=12, weight=FontWeight.BOLD),
            self.caption_field,
            Row([
                self.save_caption_btn,
                Container(expand=True),
                self.delete_image_btn,
            ]),
        ], spacing=10, scroll=ScrollMode.AUTO, expand=True)
    
    def _on_caption_change(self, e):
        """Track caption changes."""
        self.editing_caption = True
    
    def _save_caption(self, e):
        """Save the caption to the .txt file."""
        if not self.selected_image:
            return
        
        img_path = Path(self.selected_image["path"])
        caption_path = img_path.with_suffix(".txt")
        caption = self.caption_field.value.strip()
        
        try:
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Update in-memory data
            self.selected_image["caption"] = caption
            for img in self.images:
                if img["path"] == self.selected_image["path"]:
                    img["caption"] = caption
                    break
            
            self._build_image_grid()
            
            self.page.snack_bar = SnackBar(content=Text("✅ Caption saved!"))
            self.page.snack_bar.open = True
            self.editing_caption = False
            
        except Exception as ex:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Error saving caption: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
        
        self.page.update()
    
    def _pick_images(self, e):
        """Open file picker to select images."""
        self.file_picker.pick_files(
            dialog_title="Select training images",
            allowed_extensions=["png", "jpg", "jpeg", "webp", "bmp"],
            allow_multiple=True,
        )
    
    def _on_files_picked(self, e: FilePickerResultEvent):
        """Handle picked files."""
        if not e.files or not self.selected_dataset:
            return
        
        images_dir = DATASETS_DIR / self.selected_dataset / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        imported = 0
        for file in e.files:
            try:
                src_path = Path(file.path)
                dest_path = images_dir / src_path.name
                
                # Handle duplicates
                counter = 1
                while dest_path.exists():
                    stem = src_path.stem
                    dest_path = images_dir / f"{stem}_{counter}{src_path.suffix}"
                    counter += 1
                
                shutil.copy2(src_path, dest_path)
                
                # Create empty caption file
                caption_path = dest_path.with_suffix(".txt")
                caption_path.touch()
                
                imported += 1
            except Exception as ex:
                logger.error(f"Error importing {file.name}: {ex}")
        
        if imported > 0:
            self.page.snack_bar = SnackBar(content=Text(f"✅ Imported {imported} image(s)!"))
            self.page.snack_bar.open = True
            self._refresh()
        
        self.page.update()
    
    def _show_create_dialog(self, e):
        """Show dialog to create a new dataset."""
        name_input = TextField(label="Dataset Name", autofocus=True)
        
        def do_create(e):
            name = name_input.value.strip()
            if not name:
                return
            
            # Sanitize name
            name = "".join(c for c in name if c.isalnum() or c in "_ -")
            
            # Create dataset
            dataset_path = DATASETS_DIR / name
            if dataset_path.exists():
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Dataset '{name}' already exists!"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
                dialog.open = False
                self.page.update()
                return
            
            dataset_path.mkdir(parents=True)
            (dataset_path / "images").mkdir()
            
            # Create metadata
            metadata = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "images": []
            }
            self._save_dataset_info(name, metadata)
            
            dialog.open = False
            self.selected_dataset = name
            self._refresh()
            
            self.page.snack_bar = SnackBar(content=Text(f"✅ Created dataset: {name}"))
            self.page.snack_bar.open = True
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Create New Dataset"),
            content=Container(
                content=Column([
                    Text("Enter a name for the new training dataset:", size=12),
                    name_input,
                ], spacing=10),
                width=350,
            ),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton("Create", on_click=do_create),
            ],
        )
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _confirm_delete_dataset(self, e):
        """Confirm and delete the selected dataset."""
        if not self.selected_dataset:
            return
        
        def do_delete(e):
            try:
                shutil.rmtree(DATASETS_DIR / self.selected_dataset)
                
                self.page.snack_bar = SnackBar(
                    content=Text(f"✅ Deleted dataset: {self.selected_dataset}")
                )
                self.page.snack_bar.open = True
                
                self.selected_dataset = None
                self.selected_image = None
                self._refresh()
                
            except Exception as ex:
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Error: {ex}"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
            
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Dataset?"),
            content=Text(f"Are you sure you want to delete '{self.selected_dataset}'?\n\nThis will permanently delete all images and captions!"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton("Delete", bgcolor=Colors.RED_700, on_click=do_delete),
            ],
        )
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _confirm_delete_image(self, e):
        """Confirm and delete the selected image."""
        if not self.selected_image:
            return
        
        def do_delete(e):
            try:
                img_path = Path(self.selected_image["path"])
                caption_path = img_path.with_suffix(".txt")
                
                if img_path.exists():
                    img_path.unlink()
                if caption_path.exists():
                    caption_path.unlink()
                
                self.page.snack_bar = SnackBar(
                    content=Text(f"✅ Deleted: {self.selected_image['filename']}")
                )
                self.page.snack_bar.open = True
                
                self.selected_image = None
                self._load_images()
                self._update_detail_panel()
                
            except Exception as ex:
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Error: {ex}"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
            
            dialog.open = False
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Delete Image?"),
            content=Text(f"Delete '{self.selected_image['filename']}' and its caption?"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                ElevatedButton("Delete", bgcolor=Colors.RED_700, on_click=do_delete),
            ],
        )
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _open_dataset_folder(self, e):
        """Open the dataset folder in file explorer."""
        if not self.selected_dataset:
            return
        
        import subprocess
        import sys
        
        folder = DATASETS_DIR / self.selected_dataset
        if sys.platform == "win32":
            subprocess.run(["explorer", str(folder)])
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])
    
    def build(self) -> Container:
        """Build the Dataset Manager tab."""
        self._load_datasets()
        
        # Header
        header = Row([
            Text("📂 Dataset Manager", size=18, weight=FontWeight.BOLD),
            Container(expand=True),
            self.search_field,
            self.new_dataset_btn,
            self.refresh_btn,
        ], alignment=MainAxisAlignment.SPACE_BETWEEN)
        
        # Dataset list panel (left)
        dataset_panel = Container(
            content=Column([
                Text("Datasets", size=14, weight=FontWeight.BOLD),
                self.dataset_list,
            ], spacing=10),
            width=250,
            padding=padding.all(10),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(10),
        )
        
        # Image grid panel (center)
        image_panel_header = Row([
            self.dataset_info_text,
            Container(expand=True),
            self.import_btn,
            self.open_folder_btn,
            self.delete_dataset_btn,
        ])
        
        # Show appropriate content based on state
        if not self.selected_dataset:
            grid_content = self.no_dataset_msg
        elif not self.images:
            grid_content = self.empty_dataset_msg
        else:
            grid_content = self.image_grid
        
        image_panel = Container(
            content=Column([
                image_panel_header,
                grid_content,
            ], spacing=10, expand=True),
            expand=True,
            padding=padding.all(10),
        )
        
        # Main layout
        return Container(
            content=Column([
                header,
                Row([
                    self.status_text,
                ]),
                Row([
                    # Left: Dataset list
                    dataset_panel,
                    # Center: Image grid
                    image_panel,
                    # Right: Detail panel
                    Container(
                        content=self.detail_panel,
                        width=300,
                    ),
                ], expand=True, spacing=15),
                self.file_picker,  # Hidden file picker
            ], spacing=10),
            padding=padding.all(15),
            expand=True,
        )
