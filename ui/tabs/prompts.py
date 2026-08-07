"""
Prompt Library Tab - manages saved prompts using SQLite database
"""
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import logging

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Card, TextField, ElevatedButton,
    IconButton, ListView, Colors, Icons, MainAxisAlignment, ScrollMode,
    FontWeight, padding, border_radius, SnackBar, dropdown, Dropdown,
)

logger = logging.getLogger(__name__)

# Path to old YAML file for migration
YAML_PROMPTS_FILE = Path(__file__).parent.parent.parent / "config" / "saved_prompts.yaml"


class PromptLibraryTab:
    """Manage saved prompts using SQLite database."""
    
    def __init__(self, page: Page, on_use_prompt: Optional[Callable] = None, on_switch_tab: Optional[Callable] = None):
        self.page = page
        self.on_use_prompt = on_use_prompt
        self.on_switch_tab = on_switch_tab
        self.filtered_prompts = []
        self._init_db()
        self._migrate_yaml_prompts()
        self._build_ui()
    
    def _init_db(self):
        """Initialize database connection."""
        try:
            from src.database.chat_history import ChatHistoryDB
            self.db = ChatHistoryDB()  # uses core.get_effective_db_path(); see Settings > Database Location
        except ImportError:
            logger.error("ChatHistoryDB not available")
            self.db = None
    
    def _migrate_yaml_prompts(self):
        """Migrate prompts from YAML file to database (one-time migration)."""
        if not self.db or not YAML_PROMPTS_FILE.exists():
            return
        
        try:
            import yaml
            with open(YAML_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            prompts = data.get("prompts", [])
            if not prompts:
                return
            
            migrated_count = 0
            for p in prompts:
                name = p.get("name", "Untitled")
                prompt = p.get("prompt", "")
                negative = p.get("negative", "")
                
                # Handle both 'category' (string) and 'categories' (list)
                categories = p.get("categories", [])
                if not categories and p.get("category"):
                    categories = [p.get("category")]
                
                # Get settings if present
                settings = p.get("settings", {})
                steps = settings.get("steps", 25)
                guidance = settings.get("guidance", 7.5)
                width = settings.get("width", 512)
                height = settings.get("height", 512)
                lora = settings.get("lora")
                lora_strength = settings.get("lora_strength", 0.8)
                
                # Check if prompt already exists in database
                existing = self.db.get_prompt_by_name(name)
                if not existing:
                    self.db.save_prompt(
                        name=name,
                        prompt=prompt,
                        negative_prompt=negative,
                        categories=categories,
                        steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        lora=lora if lora != "None" else None,
                        lora_strength=lora_strength,
                    )
                    migrated_count += 1
            
            if migrated_count > 0:
                logger.info(f"Migrated {migrated_count} prompts from YAML to database")
                # Rename the YAML file to indicate migration complete
                backup_path = YAML_PROMPTS_FILE.with_suffix('.yaml.migrated')
                YAML_PROMPTS_FILE.rename(backup_path)
                logger.info(f"Renamed {YAML_PROMPTS_FILE} to {backup_path}")
        except Exception as e:
            logger.error(f"Error migrating YAML prompts: {e}")
    
    def _build_ui(self):
        self.prompt_list = ListView(spacing=10, padding=10, expand=True)
        self.search_field = TextField(
            label="🔍 Search prompts", 
            on_change=self._on_search_change, 
            width=300,
        )
        self.name_input = TextField(label="Prompt Name", expand=True, on_change=self._on_form_change)
        self.prompt_input = TextField(label="Prompt Text", multiline=True, min_lines=6, expand=True, on_change=self._on_form_change)
        self.negative_input = TextField(label="Negative Prompt (optional)", multiline=True, min_lines=4, expand=True)
        
        self.category_tags = {"General": False, "Portrait": False, "Landscape": False, "Fantasy": False, "Anime": False, "Other": False}
        self.category_buttons = {}
        self.category_row = Row(spacing=5, wrap=True)
        for cat in self.category_tags.keys():
            btn = ElevatedButton(cat, on_click=lambda e, c=cat: self._toggle_category(c), bgcolor=Colors.GREY_700)
            self.category_buttons[cat] = btn
            self.category_row.controls.append(btn)
        
        self.save_btn = ElevatedButton("Save Prompt", icon=Icons.SAVE, on_click=self._save_prompt, disabled=True)
        self.refresh_btn = IconButton(icon=Icons.REFRESH, tooltip="Refresh list", on_click=lambda e: self._load_prompts())
    
    def _toggle_category(self, category: str):
        self.category_tags[category] = not self.category_tags[category]
        btn = self.category_buttons[category]
        btn.bgcolor = Colors.PRIMARY if self.category_tags[category] else Colors.GREY_700
        self._update_save_btn_state()
        self.page.update()
    
    def _get_selected_categories(self) -> List[str]:
        return [cat for cat, selected in self.category_tags.items() if selected]
    
    def _on_search_change(self, e):
        self._filter_and_display_prompts()
    
    def _filter_and_display_prompts(self):
        search_text = self.search_field.value.lower().strip() if self.search_field.value else ""
        
        if not self.db:
            self.filtered_prompts = []
        else:
            all_prompts = self.db.get_all_prompts()
            
            if search_text:
                self.filtered_prompts = [
                    p for p in all_prompts 
                    if search_text in p.get("name", "").lower() or search_text in p.get("prompt", "").lower()
                ]
            else:
                self.filtered_prompts = all_prompts
        
        self._display_prompts()
    
    def _update_save_btn_state(self):
        has_name = self.name_input.value and self.name_input.value.strip()
        has_prompt = self.prompt_input.value and self.prompt_input.value.strip()
        self.save_btn.disabled = not (has_name and has_prompt)
        try:
            self.page.update()
        except:
            pass
    
    def _on_form_change(self, e):
        self._update_save_btn_state()
    
    def _load_prompts(self):
        self.search_field.value = ""
        self._filter_and_display_prompts()
    
    def _display_prompts(self):
        self.prompt_list.controls.clear()
        
        if not self.filtered_prompts:
            self.prompt_list.controls.append(
                Container(
                    content=Text("No prompts found. Add one below!", color=Colors.GREY_500, italic=True),
                    padding=padding.all(20)
                )
            )
            self.page.update()
            return
        
        for prompt in self.filtered_prompts:
            prompt_id = prompt.get("id")
            name = prompt.get("name", "Untitled")
            text = prompt.get("prompt", "")
            negative = prompt.get("negative_prompt", "")
            categories = prompt.get("categories", [])
            
            # Build category tags
            cat_text = ", ".join(categories) if categories else "No category"
            
            card = Card(
                content=Container(
                    content=Column([
                        Row([
                            Text(name, weight=FontWeight.BOLD, size=14, expand=True),
                            Text(cat_text, size=10, color=Colors.GREY_500),
                        ]),
                        Text(
                            text[:100] + "..." if len(text) > 100 else text, 
                            size=11, 
                            color=Colors.GREY_400
                        ),
                        Row([
                            ElevatedButton(
                                "Use", 
                                icon=Icons.PLAY_ARROW, 
                                on_click=lambda e, p=text, n=negative: self._use_prompt(p, n)
                            ),
                            IconButton(
                                icon=Icons.CONTENT_COPY, 
                                tooltip="Copy prompt", 
                                on_click=lambda e, p=text: self._copy_prompt(p)
                            ),
                            IconButton(
                                icon=Icons.DELETE_OUTLINE, 
                                tooltip="Delete", 
                                icon_color=Colors.RED_400, 
                                on_click=lambda e, pid=prompt_id: self._delete_prompt(pid)
                            ),
                        ], spacing=5),
                    ], spacing=4, tight=True),
                    padding=padding.all(8),
                ),
                margin=0,
            )
            self.prompt_list.controls.append(card)
        
        self.page.update()
    
    def _use_prompt(self, prompt: str, negative: str):
        if self.on_use_prompt:
            self.on_use_prompt(prompt, negative)
        if self.on_switch_tab:
            self.on_switch_tab(0)
        self.page.snack_bar = SnackBar(content=Text("✅ Prompt loaded!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _copy_prompt(self, prompt: str):
        self.page.set_clipboard(prompt)
        self.page.snack_bar = SnackBar(content=Text("Prompt copied!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _save_prompt(self, e):
        if not self.db:
            self.page.snack_bar = SnackBar(content=Text("❌ Database not available"))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        name = self.name_input.value.strip()
        prompt = self.prompt_input.value.strip()
        negative = self.negative_input.value.strip() if self.negative_input.value else ""
        categories = self._get_selected_categories() or ["General"]
        
        try:
            self.db.save_prompt(
                name=name,
                prompt=prompt,
                negative_prompt=negative,
                categories=categories,
            )
            
            # Clear form
            self.name_input.value = ""
            self.prompt_input.value = ""
            self.negative_input.value = ""
            for cat in self.category_tags:
                self.category_tags[cat] = False
                self.category_buttons[cat].bgcolor = Colors.GREY_700
            self._update_save_btn_state()
            
            self.page.snack_bar = SnackBar(content=Text(f"✅ Saved: {name}"))
            self.page.snack_bar.open = True
            self._load_prompts()
        except Exception as ex:
            self.page.snack_bar = SnackBar(content=Text(f"❌ Error: {ex}"))
            self.page.snack_bar.open = True
            self.page.update()
    
    def _delete_prompt(self, prompt_id: int):
        if not self.db:
            return
        
        try:
            prompt = self.db.get_prompt_by_id(prompt_id)
            name = prompt.get("name", "prompt") if prompt else "prompt"
            
            self.db.delete_prompt(prompt_id)
            
            self.page.snack_bar = SnackBar(content=Text(f"Deleted: {name}"))
            self.page.snack_bar.open = True
            self._load_prompts()
        except Exception as ex:
            self.page.snack_bar = SnackBar(content=Text(f"❌ Error: {ex}"))
            self.page.snack_bar.open = True
            self.page.update()
    
    def add_prompt_from_generator(self, prompt: str, negative: str, settings: dict = None):
        """Add a prompt from the image generator (called via callback)."""
        if not self.db:
            return
        
        from datetime import datetime
        words = prompt.split()[:5]
        name = " ".join(words) + ("..." if len(words) >= 5 else "")
        name = f"{name} ({datetime.now().strftime('%H:%M')})"
        
        try:
            self.db.save_prompt(
                name=name,
                prompt=prompt,
                negative_prompt=negative,
                categories=["General"],
                steps=settings.get("steps", 25) if settings else 25,
                guidance_scale=settings.get("guidance", 7.5) if settings else 7.5,
                width=settings.get("width", 512) if settings else 512,
                height=settings.get("height", 512) if settings else 512,
                lora=settings.get("lora") if settings else None,
                lora_strength=settings.get("lora_strength", 0.8) if settings else 0.8,
            )
            
            self.page.snack_bar = SnackBar(content=Text(f"💾 Saved to library: {name[:30]}..."))
            self.page.snack_bar.open = True
            self._load_prompts()
        except Exception as ex:
            logger.error(f"Error saving prompt from generator: {ex}")
    
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
                        Text("Categories:", size=12, color=Colors.GREY_400),
                        self.category_row,
                        self.prompt_input,
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
