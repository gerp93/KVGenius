"""
Model Manager Tab - download and manage models
"""
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import logging
import threading
import shutil

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Card, ElevatedButton,
    IconButton, ListView, Colors, Icons, MainAxisAlignment, ScrollMode,
    FontWeight, padding, border_radius, SnackBar, ProgressBar, ProgressRing,
    ControlState, ButtonStyle, Tabs, Tab,
)

from ui.state import app_state
from core import (
    get_available_image_models,
    get_available_chat_models,
    is_model_downloaded,
    is_chat_model_downloaded,
    download_model,
    download_chat_model,
)

logger = logging.getLogger(__name__)


class ModelManagerTab:
    """Model Manager - view and download all models (image + chat)."""
    
    def __init__(self, page: Page):
        self.page = page
        self._downloading: Dict[str, bool] = {}  # Track downloads in progress
        self.model_cache = Path(__file__).parent.parent.parent / "data" / "model_cache"
        self._build_ui()
    
    def _build_ui(self):
        self.image_model_list = ListView(spacing=10, padding=20, expand=True)
        self.chat_model_list = ListView(spacing=10, padding=20, expand=True)
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
    
    def _delete_model(self, model_name: str, repo_id: str, model_type: str):
        """Delete a downloaded model."""
        safe_name = "models--" + repo_id.replace("/", "--")
        model_path = self.model_cache / safe_name
        
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
                self.page.snack_bar = SnackBar(content=Text(f"🗑️ Deleted: {model_name}"))
                self.page.snack_bar.open = True
                self._load_models()
            except Exception as e:
                self.page.snack_bar = SnackBar(content=Text(f"❌ Error deleting: {e}"))
                self.page.snack_bar.open = True
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
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "image")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(self.image_model_list, name, info, "image", is_loaded, downloaded, is_downloading, is_hf, repo_id)
        
        # Chat Models
        chat_models = get_available_chat_models()
        for name, info in chat_models.items():
            repo_id = info.get("id", "")
            downloaded = is_chat_model_downloaded(repo_id)
            is_loaded = (app_state.loaded_model == name and app_state.loaded_model_type == "chat")
            is_downloading = repo_id in self._downloading and self._downloading[repo_id]
            
            self._add_model_card(self.chat_model_list, name, info, "chat", is_loaded, downloaded, is_downloading, True, repo_id)
    
    def _add_model_card(self, target_list: ListView, name: str, info: Dict, model_type: str, is_loaded: bool, 
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
        
        if is_hf and not is_downloading:
            if not downloaded:
                action_buttons.append(
                    ElevatedButton(
                        "Download",
                        icon=Icons.DOWNLOAD,
                        on_click=lambda e, n=name, r=repo_id, t=model_type: self._download_model(n, r, t),
                    )
                )
            else:
                # Downloaded - show re-download and delete
                action_buttons.append(
                    ElevatedButton(
                        "Re-download",
                        icon=Icons.DOWNLOAD,
                        on_click=lambda e, n=name, r=repo_id, t=model_type: self._download_model(n, r, t),
                    )
                )
                action_buttons.append(
                    ElevatedButton(
                        "Delete",
                        icon=Icons.DELETE_OUTLINE,
                        on_click=lambda e, n=name, r=repo_id, t=model_type: self._delete_model(n, r, t),
                        style=ButtonStyle(color=Colors.RED_400),
                    )
                )
        elif is_downloading:
            action_buttons.append(
                Row([
                    ProgressRing(width=16, height=16, stroke_width=2),
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
        target_list.controls.append(card)
    
    def build(self) -> Container:
        self._load_models()
        
        # Create tabs for image and chat models
        tabs = Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                Tab(
                    text="🎨 Image Models",
                    content=self.image_model_list,
                ),
                Tab(
                    text="💬 Chat Models",
                    content=self.chat_model_list,
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
                    Text("🟢 Loaded | 🟡 Downloaded | ⚪ Not Downloaded", size=11, color=Colors.GREY_500),
                    self.status_text,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Container(height=10),
                tabs,
            ]),
            padding=padding.all(15),
            expand=True,
        )
