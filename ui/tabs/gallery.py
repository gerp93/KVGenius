"""
Gallery Tab for KVGenius Desktop App.
Temporarily imports from the monolithic desktop_app.py during refactoring.
"""
# This file will be populated with the GalleryTab class
# For now, import from the original location
# TODO: Move GalleryTab class here

from typing import List, Dict, Set
import sys
from pathlib import Path
import logging

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Card, ListView, GridView,
    Image, IconButton, TextButton, ElevatedButton, Checkbox, Dropdown,
    AlertDialog, Colors, Icons, MainAxisAlignment, CrossAxisAlignment,
    ScrollMode, FontWeight, padding, border_radius, border, Stack,
    alignment, ControlState, ButtonStyle, dropdown, SnackBar, Divider,
)

from ui.state import app_state
from core import get_generated_images

logger = logging.getLogger(__name__)


class GalleryTab:
    """Gallery Tab - view generated images with list and grid views."""
    
    GRID_COLUMNS = 4
    VIEW_LIST = "list"
    VIEW_GRID = "grid"
    PAGE_SIZE_OPTIONS = [10, 20, 50, 100]
    
    def __init__(self, page: Page):
        self.page = page
        self.current_page = 0
        self.page_size = 20
        self.total_pages = 1
        self.all_images: List[Dict] = []
        self.selected_images: Set[str] = set()
        self.current_view = self.VIEW_LIST
        self.current_dialog = None  # Track current fullscreen dialog
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
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
        
        self.page_size_dropdown = Dropdown(
            label="Per Page",
            width=100,
            options=[dropdown.Option(str(size)) for size in self.PAGE_SIZE_OPTIONS],
            value="20",
            on_change=self._on_page_size_change,
        )
        
        self.total_count_text = Text("Total: 0 images", size=12, color=Colors.GREY_400)
        self.image_list = ListView(spacing=10, padding=padding.all(10), expand=True)
        
        self.image_grid = GridView(
            runs_count=self.GRID_COLUMNS,
            max_extent=200,
            child_aspect_ratio=0.85,
            spacing=10,
            run_spacing=10,
            expand=True,
            padding=padding.all(10),
        )
        
        self.view_container = Container(content=self.image_list, expand=True)
        
        self.page_label = Text("Page 1 of 1", size=12)
        self.prev_btn = IconButton(
            icon=Icons.CHEVRON_LEFT,
            tooltip="Previous page",
            on_click=self._prev_page,
            disabled=True,
        )
        self.next_btn = IconButton(
            icon=Icons.CHEVRON_RIGHT,
            tooltip="Next page",
            on_click=self._next_page,
            disabled=True,
        )
        self.pagination_row = Row([
            self.prev_btn,
            self.page_label,
            self.page_size_dropdown,
            self.next_btn,
        ], alignment=MainAxisAlignment.CENTER)
        
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
        self.selection_count = Text("", size=12, color=Colors.GREY_400)
    
    def _switch_view(self, view_type: str):
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
        self.current_page = 0
        self.selected_images.clear()
        self._load_all_images()
        if self.current_view == self.VIEW_LIST:
            self._update_list()
        else:
            self._update_grid()
        self.page.update()
    
    def _on_page_size_change(self, e):
        self.page_size = int(self.page_size_dropdown.value)
        self.current_page = 0
        self._recalculate_pages()
        self._update_list()
        self.page.update()
    
    def _load_all_images(self):
        self.all_images = get_generated_images(limit=500)
        self._recalculate_pages()
        self.total_count_text.value = f"Total: {len(self.all_images)} images"
    
    def _recalculate_pages(self):
        self.total_pages = max(1, (len(self.all_images) + self.page_size - 1) // self.page_size)
    
    def _get_current_page_images(self) -> List[Dict]:
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.all_images[start:end]
    
    def _prev_page(self, e=None):
        if self.current_page > 0:
            self.current_page -= 1
            self._update_list()
            self.page.update()
    
    def _next_page(self, e=None):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_list()
            self.page.update()
    
    def _toggle_select_all(self, e=None):
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
    
    def _toggle_selection(self, filepath: str):
        if filepath in self.selected_images:
            self.selected_images.discard(filepath)
        else:
            self.selected_images.add(filepath)
        self._update_selection_ui()
        if self.current_view == self.VIEW_LIST:
            self._update_list()
        else:
            self._update_grid()
        self.page.update()
    
    def _update_selection_ui(self):
        count = len(self.selected_images)
        self.selection_count.value = f"{count} selected" if count > 0 else ""
        self.delete_selected_btn.visible = count > 0
    
    def _delete_selected(self, e=None):
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
            content=Text(f"Delete {count} selected images?"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Delete All", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _open_folder(self, filepath: str):
        import subprocess
        folder = Path(filepath).parent
        if sys.platform == "win32":
            subprocess.run(["explorer", str(folder)])
    
    def _copy_prompt(self, prompt: str):
        self.page.set_clipboard(prompt)
        self.page.snack_bar = SnackBar(content=Text("📋 Prompt copied!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _delete_image(self, filepath: str):
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
            content=Text(f"Delete {Path(filepath).name}?"),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Delete", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _show_fullscreen_image(self, filepath: str, img_info: dict):
        """Show image in fullscreen popup."""
        
        def close_dialog(e=None):
            """Close the dialog properly."""
            try:
                dialog.open = False
                self.page.update()
            except Exception as ex:
                logger.error(f"Error closing dialog: {ex}")
        
        close_btn = IconButton(icon=Icons.CLOSE, icon_size=24, on_click=close_dialog)
        
        fullscreen_image = Image(
            src=filepath,
            fit=ft.ImageFit.CONTAIN,
            expand=True,
        )
        
        info_controls = [
            Text(f"📁 {img_info.get('filename', 'Image')}", weight=FontWeight.BOLD, size=14),
            Text(f"🎨 Model: {img_info.get('model', 'Unknown')}", size=12),
            Text(f"🌱 Seed: {img_info.get('seed', 'N/A')}", size=12),
        ]
        
        if img_info.get('prompt'):
            info_controls.extend([
                Text("✨ Prompt:", size=12, weight=FontWeight.BOLD),
                Text(img_info.get('prompt', ''), size=11, color=Colors.GREY_300, selectable=True),
            ])
        
        info_text = Column(info_controls, spacing=10, scroll=ScrollMode.AUTO)
        
        def handle_delete(e):
            close_dialog()
            self._delete_image(filepath)
        
        buttons = Row([
            IconButton(icon=Icons.CONTENT_COPY, tooltip="Copy Prompt",
                      on_click=lambda e: self._copy_prompt(img_info.get('prompt', ''))),
            IconButton(icon=Icons.FOLDER_OPEN, tooltip="Open Folder",
                      on_click=lambda e: self._open_folder(filepath)),
            IconButton(icon=Icons.DELETE_OUTLINE, tooltip="Delete", icon_color=Colors.RED_400,
                      on_click=handle_delete),
        ], spacing=10)
        
        content = Row([
            Container(content=fullscreen_image, expand=2, bgcolor=Colors.with_opacity(0.3, Colors.BLACK)),
            Container(
                content=Column([
                    Row([Text("Image Details", weight=FontWeight.BOLD), Container(expand=True), close_btn]),
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
            on_dismiss=close_dialog,
        )
        
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _close_fullscreen(self):
        """Close fullscreen dialog - deprecated, using inline close now."""
        pass
    
    def _handle_fullscreen_delete(self, filepath: str):
        """Handle deletion from fullscreen view - deprecated, using inline handler now."""
        self._delete_image(filepath)
    
    def _update_list(self):
        self.image_list.controls.clear()
        page_images = self._get_current_page_images()
        
        if not page_images:
            self.image_list.controls.append(Text("No generated images yet", color=Colors.GREY_500))
        else:
            for img_info in page_images:
                filepath = img_info["path"]
                prompt = img_info.get("prompt", "")
                is_selected = filepath in self.selected_images
                
                actions = Row([
                    Checkbox(value=is_selected, on_change=lambda e, p=filepath: self._toggle_selection(p)),
                    IconButton(icon=Icons.FULLSCREEN, tooltip="View Fullscreen", icon_size=18,
                              on_click=lambda e, p=filepath, info=img_info: self._show_fullscreen_image(p, info)),
                    IconButton(icon=Icons.FOLDER_OPEN, tooltip="Open Folder", icon_size=18,
                              on_click=lambda e, p=filepath: self._open_folder(p)),
                    IconButton(icon=Icons.CONTENT_COPY, tooltip="Copy Prompt", icon_size=18,
                              on_click=lambda e, pr=prompt: self._copy_prompt(pr)),
                    IconButton(icon=Icons.DELETE_OUTLINE, tooltip="Delete", icon_size=18,
                              icon_color=Colors.RED_400, on_click=lambda e, p=filepath: self._delete_image(p)),
                ], spacing=0)
                
                card = Card(
                    content=Container(
                        content=Row([
                            Image(src=filepath, width=120, height=120, fit=ft.ImageFit.COVER,
                                 border_radius=border_radius.all(8)),
                            Column([
                                Text(img_info["filename"], weight=FontWeight.BOLD, size=13),
                                Text(f"Model: {img_info['model']}", size=11, color=Colors.GREY_400),
                                Text(f"Seed: {img_info['seed']}", size=11, color=Colors.GREY_400),
                                actions,
                            ], expand=True, spacing=3),
                        ], spacing=15),
                        padding=padding.all(10),
                        bgcolor=Colors.with_opacity(0.2, Colors.PRIMARY) if is_selected else None,
                    ),
                )
                self.image_list.controls.append(card)
        
        self.page_label.value = f"Page {self.current_page + 1} of {self.total_pages}"
        self.prev_btn.disabled = self.current_page <= 0
        self.next_btn.disabled = self.current_page >= self.total_pages - 1
        self._update_selection_ui()
    
    def _update_grid(self):
        self.image_grid.controls.clear()
        
        if not self.all_images:
            self.image_grid.controls.append(
                Container(content=Text("No generated images yet", color=Colors.GREY_500), alignment=alignment.center)
            )
        else:
            for img_info in self.all_images:
                filepath = img_info["path"]
                is_selected = filepath in self.selected_images
                
                card = Container(
                    content=Stack([
                        Container(
                            content=Image(src=filepath, fit=ft.ImageFit.COVER, border_radius=border_radius.all(8)),
                            border_radius=border_radius.all(8),
                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                        ),
                        Container(
                            bgcolor=Colors.with_opacity(0.3, Colors.PRIMARY) if is_selected else None,
                            border=border.all(3, Colors.PRIMARY) if is_selected else None,
                            border_radius=border_radius.all(8),
                        ),
                        Container(
                            content=Checkbox(value=is_selected, on_change=lambda e, p=filepath: self._toggle_selection(p)),
                            alignment=alignment.top_left,
                            padding=padding.only(left=5, top=5),
                        ),
                        Container(
                            content=Container(
                                content=Row([
                                    IconButton(icon=Icons.FULLSCREEN, tooltip="Fullscreen", icon_size=16, icon_color=Colors.WHITE,
                                              on_click=lambda e, p=filepath, info=img_info: self._show_fullscreen_image(p, info)),
                                    IconButton(icon=Icons.CONTENT_COPY, tooltip="Copy Prompt", icon_size=16, icon_color=Colors.WHITE,
                                              on_click=lambda e, pr=img_info.get("prompt", ""): self._copy_prompt(pr)),
                                    IconButton(icon=Icons.DELETE_OUTLINE, tooltip="Delete", icon_size=16, icon_color=Colors.RED_300,
                                              on_click=lambda e, p=filepath: self._delete_image(p)),
                                ], spacing=0, alignment=MainAxisAlignment.CENTER),
                                bgcolor=Colors.with_opacity(0.7, Colors.BLACK),
                                border_radius=border_radius.only(bottom_left=8, bottom_right=8),
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
        self._load_all_images()
        self._update_list()
        
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
