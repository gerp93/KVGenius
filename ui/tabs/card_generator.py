"""
Card Generator Tab for KVGenius Flet App
Generate CAH-style party game cards using text models.
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField, 
    ElevatedButton, IconButton, TextButton, Card,
    Icons, Colors, FontWeight, ScrollMode, AlertDialog,
    SnackBar, ListView, Dropdown, dropdown, Slider,
    ProgressBar, ProgressRing, Divider, Checkbox, Tabs, Tab,
    padding, border_radius, alignment, GridView,
    ControlState, ButtonStyle, MainAxisAlignment, CrossAxisAlignment,
    DataTable, DataColumn, DataRow, DataCell,
)

logger = logging.getLogger(__name__)

# Import card generator components
try:
    from src.cards.cah_generator import (
        CAHGenerator, CardType, CardStyle, Card as CAHCard,
        get_generation_prompt, SYSTEM_PROMPTS,
    )
    CAH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CAH generator not available: {e}")
    CAH_AVAILABLE = False

# Import app state for model access
try:
    from ui.state import app_state
except ImportError:
    app_state = None


class CardGeneratorTab:
    """
    Card Generator Tab - Generate CAH-style party game cards.
    Uses the loaded chat model to generate black (prompt) and white (answer) cards.
    """
    
    # Style choices with emojis
    STYLES = {
        "classic": ("🎭 Classic", "Irreverent & crude humor"),
        "absurd": ("🌀 Absurd", "Surreal & nonsensical"),
        "dark": ("🖤 Dark", "Gallows humor & morbid"),
        "wholesome": ("✨ Wholesome", "Family-friendly"),
        "nerdy": ("🎮 Nerdy", "Gaming, tech, pop culture"),
        "custom": ("✏️ Custom", "Define your own style"),
    }
    
    # Card type choices
    CARD_TYPES = {
        "both": ("🃏 Both Types", "Black & white cards"),
        "black": ("⬛ Black Cards", "Prompts with blanks"),
        "white": ("⬜ White Cards", "Answer cards"),
    }
    
    def __init__(self, page: Page):
        self.page = page
        self.cah_generator: Optional[CAHGenerator] = None
        self.generating = False
        self.generated_cards: List[Dict] = []
        
        # Selection tracking for multi-delete
        self.selected_generated_cards: set = set()  # Indices of selected cards in generated output
        self.selected_library_cards: set = set()    # Card IDs of selected cards in library
        
        # File picker for exports
        self.file_picker = ft.FilePicker(on_result=self._on_export_file_picked)
        self.page.overlay.append(self.file_picker)
        self._pending_export_format = None
        
        # Initialize CAH generator
        if CAH_AVAILABLE:
            try:
                self.cah_generator = CAHGenerator()
            except Exception as e:
                logger.error(f"Failed to init CAH generator: {e}")
        
        self._build_ui()
    
    def _build_ui(self):
        """Build all UI components."""
        
        # ============ GENERATE TAB COMPONENTS ============
        
        # Topic input
        self.topic_input = TextField(
            label="📝 Topic / Theme",
            hint_text="e.g., Office life, Dating apps, Video games...",
            multiline=True,
            min_lines=2,
            max_lines=3,
        )
        
        # Card type dropdown
        self.card_type_dropdown = Dropdown(
            label="🃏 Card Type",
            width=250,
            options=[
                dropdown.Option(key, f"{self.CARD_TYPES[key][0]}")
                for key in self.CARD_TYPES.keys()
            ],
            value="both",
        )
        
        # Style dropdown
        self.style_dropdown = Dropdown(
            label="🎨 Style",
            width=250,
            options=[
                dropdown.Option(key, f"{self.STYLES[key][0]} - {self.STYLES[key][1]}")
                for key in self.STYLES.keys()
            ],
            value="classic",
            on_change=self._on_style_change,
        )
        
        # Custom style input (hidden by default)
        self.custom_style_input = TextField(
            label="✏️ Custom Style Instructions",
            hint_text="Describe your custom style...",
            multiline=True,
            min_lines=2,
            max_lines=4,
            visible=False,
        )
        
        # LoRA dropdown for card generation LoRAs
        self.lora_dropdown = Dropdown(
            label="🧬 Card LoRA",
            width=250,
            options=[dropdown.Option("none", "None (Base Model)")],
            value="none",
            hint_text="Select a trained card LoRA",
        )
        self.lora_refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh LoRA list",
            on_click=self._refresh_lora_list,
        )
        # Defer LoRA list loading to avoid initialization issues
        # Will be loaded when build() is called or refresh is clicked
        
        # Quantity slider
        self.quantity_value = Text("10", size=12, weight=FontWeight.BOLD)
        self.quantity_slider = Slider(
            min=1, max=20, value=10,
            divisions=19,
            expand=True,
            on_change=self._on_quantity_change,
        )
        
        # Generate button
        self.generate_btn = ElevatedButton(
            "🎴 Generate Cards",
            icon=Icons.AUTO_AWESOME,
            bgcolor=Colors.GREEN_700,
            color=Colors.WHITE,
            on_click=self._generate_cards,
        )
        
        # Status and progress
        self.status_text = Text("", size=12)
        self.progress_bar = ProgressBar(visible=False)
        
        # No model banner
        self.no_model_banner = Container(
            content=Row([
                ft.Icon(Icons.WARNING_AMBER_ROUNDED, size=24, color=Colors.AMBER_400),
                Text("💬 Load a chat model from the header to generate cards", 
                     size=14, color=Colors.AMBER_300),
            ], spacing=10, alignment=MainAxisAlignment.CENTER),
            bgcolor=Colors.with_opacity(0.3, Colors.AMBER_900),
            padding=padding.all(12),
            border_radius=border_radius.all(8),
            visible=True,
        )
        
        # Generated cards output (ListView for scrollable list)
        self.cards_output = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Selection controls for generated cards
        self.gen_selection_count = Text("", size=11, color=Colors.GREY_400)
        self.delete_selected_gen_btn = ElevatedButton(
            "🗑️ Delete Selected (0)",
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._delete_selected_generated,
            visible=False,
        )
        
        # ============ FROM ARTICLE TAB COMPONENTS ============
        
        self.article_input = TextField(
            label="📄 Article Text",
            hint_text="Paste article text here...\n\nThe AI will extract the funniest jokes and phrases.",
            multiline=True,
            min_lines=10,
            max_lines=15,
        )
        
        self.article_card_type = Dropdown(
            label="🃏 Card Type",
            width=250,
            options=[
                dropdown.Option(key, f"{self.CARD_TYPES[key][0]}")
                for key in self.CARD_TYPES.keys()
            ],
            value="both",
        )
        
        self.article_quantity_value = Text("8", size=12, weight=FontWeight.BOLD)
        self.article_quantity_slider = Slider(
            min=3, max=15, value=8,
            divisions=12,
            expand=True,
            on_change=self._on_article_quantity_change,
        )
        
        self.extract_btn = ElevatedButton(
            "📰 Extract Cards",
            icon=Icons.CONTENT_CUT,
            bgcolor=Colors.BLUE_700,
            color=Colors.WHITE,
            on_click=self._extract_from_article,
        )
        
        self.article_status = Text("", size=12)
        self.article_progress = ProgressBar(visible=False)
        
        self.extracted_cards_output = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # ============ MY CARDS (LIBRARY) TAB COMPONENTS ============
        
        # Pagination state
        self.library_current_page = 0
        self.library_page_size = 25
        self.library_total_pages = 1
        self.library_filtered_cards = []
        
        # Use ListView which handles its own scrolling
        self.library_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Search/filter controls
        self.library_search_field = TextField(
            label="🔍 Search cards",
            width=300,
            on_change=lambda e: self._apply_library_filters(),
        )
        
        self.library_type_filter = Dropdown(
            label="Card Type",
            width=150,
            options=[
                dropdown.Option("all", "All Types"),
                dropdown.Option("black", "⬛ Black Only"),
                dropdown.Option("white", "⬜ White Only"),
            ],
            value="all",
            on_change=lambda e: self._apply_library_filters(),
        )
        
        self.library_fav_filter = Dropdown(
            label="Favorites",
            width=150,
            options=[
                dropdown.Option("all", "All Cards"),
                dropdown.Option("favorited", "⭐ Favorites Only"),
                dropdown.Option("not_favorited", "☆ Not Favorited"),
            ],
            value="all",
            on_change=lambda e: self._apply_library_filters(),
        )
        
        self.library_exported_filter = Dropdown(
            label="Exported",
            width=150,
            options=[
                dropdown.Option("all", "All Cards"),
                dropdown.Option("exported", "✅ Exported"),
                dropdown.Option("not_exported", "⬜ Not Exported"),
            ],
            value="all",
            on_change=lambda e: self._apply_library_filters(),
        )
        
        self.library_sort = Dropdown(
            label="Sort By",
            width=180,
            options=[
                dropdown.Option("newest", "📅 Newest First"),
                dropdown.Option("oldest", "📅 Oldest First"),
                dropdown.Option("alpha_asc", "🔤 A-Z"),
                dropdown.Option("alpha_desc", "🔤 Z-A"),
                dropdown.Option("fav_first", "⭐ Favorites First"),
                dropdown.Option("fav_last", "☆ Favorites Last"),
            ],
            value="newest",
            on_change=lambda e: self._apply_library_filters(),
        )
        
        # Pagination controls
        self.library_page_label = Text("Page 1 of 1", size=12)
        self.library_prev_btn = IconButton(
            icon=Icons.CHEVRON_LEFT,
            tooltip="Previous page",
            on_click=self._library_prev_page,
            disabled=True,
        )
        self.library_next_btn = IconButton(
            icon=Icons.CHEVRON_RIGHT,
            tooltip="Next page",
            on_click=self._library_next_page,
            disabled=True,
        )
        
        self.refresh_library_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh library",
            on_click=lambda e: self._load_library(),
        )
        
        self.export_json_btn = ElevatedButton(
            "📥 Export JSON",
            on_click=lambda e: self._export_cards("json"),
        )
        
        self.export_txt_btn = ElevatedButton(
            "📥 Export TXT",
            on_click=lambda e: self._export_cards("txt"),
        )
        
        # Selection controls for library
        self.select_page_lib_btn = TextButton(
            "☑️ Select Page",
            on_click=self._select_page,
        )
        self.select_all_lib_btn = TextButton(
            "☑️ Select All Filtered",
            on_click=self._select_all_filtered,
        )
        self.clear_selection_lib_btn = TextButton(
            "☐ Clear Selection",
            on_click=self._clear_selection,
            visible=False,
        )
        self.lib_selection_count = Text("", size=11, color=Colors.GREY_400)
        self.delete_selected_lib_btn = ElevatedButton(
            "🗑️ Delete Selected (0)",
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._delete_selected_library,
            visible=False,
        )
        
        self.favorite_selected_lib_btn = ElevatedButton(
            "⭐ Favorite Selected (0)",
            bgcolor=Colors.AMBER_700,
            color=Colors.WHITE,
            on_click=self._favorite_selected_library,
            visible=False,
        )
        
        self.unfavorite_selected_lib_btn = ElevatedButton(
            "☆ Unfavorite Selected (0)",
            bgcolor=Colors.GREY_700,
            color=Colors.WHITE,
            on_click=self._unfavorite_selected_library,
            visible=False,
        )
        
        self.export_selected_lib_btn = ElevatedButton(
            "✅ Mark Exported (0)",
            bgcolor=Colors.GREEN_700,
            color=Colors.WHITE,
            on_click=self._export_selected_library,
            visible=False,
        )
        
        self.unexport_selected_lib_btn = ElevatedButton(
            "⬜ Unmark Exported (0)",
            bgcolor=Colors.BLUE_GREY_700,
            color=Colors.WHITE,
            on_click=self._unexport_selected_library,
            visible=False,
        )
        
        self.clear_all_btn = ElevatedButton(
            "🗑️ Clear All",
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._confirm_clear_all,
        )
        
        self.library_stats = Text("", size=12, color=Colors.GREY_400)
        
        # Delete last N controls
        self.delete_n_value = Text("1", size=12)
        self.delete_n_slider = Slider(
            min=1, max=20, value=1,
            divisions=19,
            width=150,
            on_change=self._on_delete_n_change,
        )
        self.delete_last_btn = ElevatedButton(
            "🗑️ Delete Last",
            bgcolor=Colors.ORANGE_700,
            color=Colors.WHITE,
            on_click=self._delete_last_n,
        )
    
    def _on_style_change(self, e):
        """Handle style dropdown change."""
        self.custom_style_input.visible = (self.style_dropdown.value == "custom")
        self.page.update()
    
    def _load_lora_list(self):
        """Load available card LoRAs into dropdown."""
        try:
            from core.chat_gen import get_available_chat_loras
            
            loras = get_available_chat_loras("cardgen")
            
            options = [dropdown.Option("none", "None (Base Model)")]
            for lora in loras:
                name = lora["name"]
                desc = lora.get("description", "")
                label = f"🧬 {name}"
                if desc:
                    label += f" - {desc[:30]}..."
                options.append(dropdown.Option(name, label))
            
            self.lora_dropdown.options = options
            
            # Reset selection if current is no longer valid
            current = self.lora_dropdown.value
            valid_keys = [opt.key for opt in options]
            if current not in valid_keys:
                self.lora_dropdown.value = "none"
                
        except Exception as ex:
            logger.error(f"Failed to load LoRA list: {ex}")
    
    def _refresh_lora_list(self, e):
        """Refresh the LoRA dropdown list."""
        self._load_lora_list()
        self.page.snack_bar = SnackBar(content=Text("🔄 LoRA list refreshed"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _on_quantity_change(self, e):
        """Handle quantity slider change."""
        self.quantity_value.value = str(int(self.quantity_slider.value))
        self.page.update()
    
    def _on_article_quantity_change(self, e):
        """Handle article quantity slider change."""
        self.article_quantity_value.value = str(int(self.article_quantity_slider.value))
        self.page.update()
    
    def _on_delete_n_change(self, e):
        """Handle delete N slider change."""
        self.delete_n_value.value = str(int(self.delete_n_slider.value))
        self.page.update()
    
    def _check_model_loaded(self) -> bool:
        """Check if a chat model is loaded."""
        if app_state and app_state.loaded_model_type == "chat":
            self.no_model_banner.visible = False
            return True
        self.no_model_banner.visible = True
        return False
    
    def _add_card_to_output(self, card_text: str, card_type: str, output_list: ListView, card_index: int = -1):
        """Add a card item to the output list with selection and delete options."""
        is_black = card_type == "black"
        emoji = "⬛" if is_black else "⬜"
        bg_color = Colors.GREY_900 if is_black else Colors.GREY_800
        
        # Checkbox for multi-select
        select_cb = Checkbox(
            value=False,
            on_change=lambda e, idx=card_index: self._toggle_generated_selection(idx, e.control.value),
        )
        
        card_item = Container(
            content=Row([
                select_cb,
                Text(emoji, size=20),
                Text(card_text, size=13, expand=True, selectable=True),
                IconButton(
                    icon=Icons.CONTENT_COPY,
                    icon_size=16,
                    tooltip="Copy",
                    on_click=lambda e, t=card_text: self._copy_to_clipboard(t),
                ),
                IconButton(
                    icon=Icons.DELETE_OUTLINE,
                    icon_size=16,
                    icon_color=Colors.RED_400,
                    tooltip="Delete this card",
                    on_click=lambda e, idx=card_index: self._delete_generated_card(idx),
                ),
            ], spacing=10),
            bgcolor=bg_color,
            padding=padding.all(12),
            border_radius=border_radius.all(8),
        )
        output_list.controls.append(card_item)
    
    def _toggle_generated_selection(self, index: int, selected: bool):
        """Toggle selection of a generated card."""
        if selected:
            self.selected_generated_cards.add(index)
        else:
            self.selected_generated_cards.discard(index)
        self._update_generated_selection_ui()
    
    def _delete_generated_card(self, index: int):
        """Delete a single generated card (before saving)."""
        if 0 <= index < len(self.generated_cards):
            del self.generated_cards[index]
            self.selected_generated_cards.discard(index)
            # Adjust indices in selection set
            self.selected_generated_cards = {i - 1 if i > index else i for i in self.selected_generated_cards if i != index}
            self._refresh_generated_output()
            self.page.snack_bar = SnackBar(content=Text("🗑️ Card removed"))
            self.page.snack_bar.open = True
            self.page.update()
    
    def _delete_selected_generated(self, e=None):
        """Delete all selected generated cards."""
        if not self.selected_generated_cards:
            return
        
        # Delete in reverse order to maintain indices
        for idx in sorted(self.selected_generated_cards, reverse=True):
            if 0 <= idx < len(self.generated_cards):
                del self.generated_cards[idx]
        
        self.selected_generated_cards.clear()
        self._refresh_generated_output()
        self.page.snack_bar = SnackBar(content=Text("🗑️ Selected cards removed"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _refresh_generated_output(self):
        """Refresh the generated cards output display."""
        self.cards_output.controls.clear()
        for i, card in enumerate(self.generated_cards):
            self._add_card_to_output(card["text"], card["type"], self.cards_output, card_index=i)
        self._update_generated_selection_ui()
        self.page.update()
    
    def _update_generated_selection_ui(self):
        """Update the selection count and delete button visibility."""
        count = len(self.selected_generated_cards)
        if hasattr(self, 'delete_selected_gen_btn'):
            self.delete_selected_gen_btn.visible = count > 0
            self.delete_selected_gen_btn.text = f"🗑️ Delete Selected ({count})"
        if hasattr(self, 'gen_selection_count'):
            self.gen_selection_count.value = f"{count} selected" if count > 0 else ""
        try:
            self.page.update()
        except:
            pass
    
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        self.page.set_clipboard(text)
        self.page.snack_bar = SnackBar(content=Text("📋 Copied to clipboard!"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _generate_cards(self, e):
        """Generate cards using the loaded chat model."""
        if self.generating:
            return
        
        if not self._check_model_loaded():
            self.page.snack_bar = SnackBar(
                content=Text("❌ Please load a chat model first"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        topic = self.topic_input.value.strip() if self.topic_input.value else ""
        if not topic:
            self.page.snack_bar = SnackBar(
                content=Text("❌ Please enter a topic or theme"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Start generation
        self.generating = True
        self.generate_btn.disabled = True
        self.progress_bar.visible = True
        self.status_text.value = "🔄 Generating cards..."
        self.cards_output.controls.clear()
        self.page.update()
        
        def do_generate():
            lora_loaded = False
            selected_lora = self.lora_dropdown.value
            
            try:
                from core.chat_gen import generate_chat_response, load_chat_lora, unload_chat_lora, get_current_lora
                
                # Load LoRA if selected and not already loaded
                if selected_lora and selected_lora != "none":
                    current_lora = get_current_lora()
                    if current_lora != selected_lora:
                        def update_status_lora():
                            self.status_text.value = f"🧬 Loading LoRA: {selected_lora}..."
                            self.page.update()
                        self.page.run_thread(update_status_lora)
                        
                        success, msg = load_chat_lora(selected_lora, "cardgen")
                        if not success:
                            raise Exception(f"Failed to load LoRA: {msg}")
                        lora_loaded = True
                        logger.info(f"Loaded LoRA for generation: {selected_lora}")
                
                # Get parameters
                card_type = self.card_type_dropdown.value
                style = self.style_dropdown.value
                quantity = int(self.quantity_slider.value)
                custom_style = self.custom_style_input.value if style == "custom" else ""
                
                # Build prompts
                style_enum = CardStyle(style)
                type_enum = CardType(card_type)
                
                system_prompt = SYSTEM_PROMPTS.get(style_enum, SYSTEM_PROMPTS[CardStyle.CLASSIC])
                if style == "custom" and custom_style:
                    system_prompt = f"{system_prompt}\n\nAdditional instructions: {custom_style}"
                
                user_prompt = get_generation_prompt(topic, type_enum, quantity, style_enum, custom_style)
                
                def update_status_gen():
                    lora_info = f" (with {selected_lora})" if lora_loaded else ""
                    self.status_text.value = f"🔄 Generating cards{lora_info}..."
                    self.page.update()
                self.page.run_thread(update_status_gen)
                
                # Generate using the chat model (no history to avoid context pollution)
                success, response = generate_chat_response(
                    user_message=user_prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=800,
                    temperature=0.8,
                    use_history=False,
                )
                
                if not success:
                    raise Exception(response)
                
                # Parse the response
                cards = self._parse_card_response(response, topic, style_enum, type_enum)
                
                # Update UI on main thread
                def update_ui():
                    self.cards_output.controls.clear()
                    self.generated_cards = cards.copy() if cards else []
                    self.selected_generated_cards.clear()
                    
                    if cards:
                        for i, card in enumerate(cards):
                            self._add_card_to_output(card["text"], card["type"], self.cards_output, card_index=i)
                        
                        # Save cards
                        if self.cah_generator:
                            self.cah_generator.add_cards([
                                CAHCard(
                                    id="",
                                    text=c["text"],
                                    card_type=c["type"],
                                    style=style,
                                    topic=topic,
                                    blanks=c.get("blanks", 0),
                                )
                                for c in cards
                            ])
                        
                        black_count = len([c for c in cards if c["type"] == "black"])
                        white_count = len([c for c in cards if c["type"] == "white"])
                        self.status_text.value = f"✅ Generated {len(cards)} cards! ({black_count} black, {white_count} white)"
                    else:
                        self.status_text.value = "⚠️ Could not parse cards from response. Try again."
                    
                    self.generating = False
                    self.generate_btn.disabled = False
                    self.progress_bar.visible = False
                    self.page.update()
                
                self.page.run_thread(update_ui)
                
            except Exception as ex:
                logger.error(f"Error generating cards: {ex}")
                error_msg = str(ex)
                
                def show_error():
                    self.status_text.value = f"❌ Error: {error_msg}"
                    self.generating = False
                    self.generate_btn.disabled = False
                    self.progress_bar.visible = False
                    self.page.update()
                
                self.page.run_thread(show_error)
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _parse_card_response(self, response: str, topic: str, style: CardStyle, card_type: CardType) -> List[Dict]:
        """Parse generated response into card objects."""
        cards = []
        
        # Try JSON parsing first
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "text" in item:
                            card_text = item["text"].strip()
                            if card_text:
                                cards.append({
                                    "text": card_text,
                                    "type": item.get("type", "white"),
                                    "blanks": item.get("blanks", 0),
                                })
                    if cards:
                        return cards[:20]  # If JSON parsing worked, return early
        except json.JSONDecodeError:
            pass
        
        # Fallback: line-by-line parsing
        if not cards:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("```"):
                    continue
                
                # Try to parse individual line as JSON object
                if line.startswith("{") and line.endswith("}"):
                    try:
                        item = json.loads(line.rstrip(","))
                        if isinstance(item, dict) and "text" in item:
                            card_text = item["text"].strip()
                            if card_text:
                                cards.append({
                                    "text": card_text,
                                    "type": item.get("type", "white"),
                                    "blanks": item.get("blanks", 0),
                                })
                                continue
                    except:
                        pass
                
                # Clean up JSON fragments from text
                line = line.strip(",[]")
                if line.startswith("{"):
                    # Try to extract text from malformed JSON
                    import re
                    text_match = re.search(r'"text"\s*:\s*"([^"]+)"', line)
                    if text_match:
                        line = text_match.group(1)
                    else:
                        continue  # Skip lines that look like broken JSON
                
                # Remove list markers
                if line and line[0] in "-*•0123456789":
                    line = line.lstrip("-*•0123456789.) ").strip()
                
                # Remove emoji prefixes
                if line.startswith(("⬛", "⬜")):
                    line = line[1:].strip()
                
                # Remove quotes
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                
                if len(line) > 5:
                    # Determine card type by presence of blanks
                    is_black = "___" in line or "_" in line
                    cards.append({
                        "text": line,
                        "type": "black" if is_black else "white",
                        "blanks": line.count("___"),
                    })
        
        return cards[:20]  # Limit to 20 cards max
    
    def _extract_from_article(self, e):
        """Extract cards from article text."""
        if self.generating:
            return
        
        if not self._check_model_loaded():
            self.page.snack_bar = SnackBar(
                content=Text("❌ Please load a chat model first"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        article = self.article_input.value.strip() if self.article_input.value else ""
        if not article or len(article) < 50:
            self.page.snack_bar = SnackBar(
                content=Text("❌ Please paste a longer article (at least 50 characters)"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Start extraction
        self.generating = True
        self.extract_btn.disabled = True
        self.article_progress.visible = True
        self.article_status.value = "🔄 Extracting cards from article..."
        self.extracted_cards_output.controls.clear()
        self.page.update()
        
        def do_extract():
            try:
                from core.chat_gen import generate_chat_response
                from src.cards.cah_generator import ARTICLE_EXTRACTION_PROMPT, get_article_extraction_prompt
                
                card_type = self.article_card_type.value
                quantity = int(self.article_quantity_slider.value)
                type_enum = CardType(card_type)
                
                user_prompt = get_article_extraction_prompt(article, type_enum, quantity)
                
                success, response = generate_chat_response(
                    user_message=user_prompt,
                    system_prompt=ARTICLE_EXTRACTION_PROMPT,
                    max_new_tokens=800,
                    temperature=0.7,
                    use_history=False,
                )
                
                if not success:
                    raise Exception(response)
                
                cards = self._parse_card_response(response, "Article extraction", CardStyle.ABSURD, type_enum)
                
                def update_ui():
                    self.extracted_cards_output.controls.clear()
                    
                    if cards:
                        for card in cards:
                            self._add_card_to_output(card["text"], card["type"], self.extracted_cards_output)
                        
                        # Save cards
                        if self.cah_generator:
                            self.cah_generator.add_cards([
                                CAHCard(
                                    id="",
                                    text=c["text"],
                                    card_type=c["type"],
                                    style="absurd",
                                    topic="Article extraction",
                                    blanks=c.get("blanks", 0),
                                )
                                for c in cards
                            ])
                        
                        black_count = len([c for c in cards if c["type"] == "black"])
                        white_count = len([c for c in cards if c["type"] == "white"])
                        self.article_status.value = f"✅ Extracted {len(cards)} cards! ({black_count} black, {white_count} white)"
                    else:
                        self.article_status.value = "⚠️ Could not extract cards. Try different content."
                    
                    self.generating = False
                    self.extract_btn.disabled = False
                    self.article_progress.visible = False
                    self.page.update()
                
                self.page.run_thread(update_ui)
                
            except Exception as ex:
                logger.error(f"Error extracting cards: {ex}")
                error_msg = str(ex)
                
                def show_error():
                    self.article_status.value = f"❌ Error: {error_msg}"
                    self.generating = False
                    self.extract_btn.disabled = False
                    self.article_progress.visible = False
                    self.page.update()
                
                self.page.run_thread(show_error)
        
        threading.Thread(target=do_extract, daemon=True).start()
    
    def _load_library(self):
        """Load and display library cards with pagination and filtering."""
        logger.info("_load_library called")
        self.library_search_field.value = ""
        self.library_type_filter.value = "all"
        self.library_fav_filter.value = "all"
        self.library_exported_filter.value = "all"
        self.library_sort.value = "newest"
        self.library_current_page = 0
        self.library_filtered_cards = []
        self.selected_library_cards.clear()  # Clear selections on full refresh
        
        if not self.cah_generator:
            logger.warning("cah_generator is None!")
            self.library_list.controls.clear()
            self.library_list.controls.append(
                Text("⚠️ Card generator not available", color=Colors.GREY_500)
            )
            self.library_stats.value = "No cards loaded"
            if self.page:
                self.page.update()
            return
        
        # Load all cards and apply filters
        self._apply_library_filters()
    
    def _apply_library_filters(self, reset_page: bool = True):
        """Apply search and type filters to the library.
        
        Args:
            reset_page: If True, reset to page 1. If False, preserve current page.
        """
        if not self.cah_generator:
            return
        
        all_cards = self.cah_generator.get_cards()
        
        if not all_cards:
            self.library_filtered_cards = []
            self.library_list.controls.clear()
            self.library_list.controls.append(
                Text("No saved cards yet. Generate some!", color=Colors.GREY_500, italic=True)
            )
            self.library_stats.value = "0 cards"
            self._update_pagination_ui()
            if self.page:
                self.page.update()
            return
        
        # Get filter values safely
        search_text = ""
        card_type_filter = "all"
        fav_filter = "all"
        exported_filter = "all"
        sort_by = "newest"
        try:
            if hasattr(self, 'library_search_field') and self.library_search_field.value:
                search_text = self.library_search_field.value.lower().strip()
            if hasattr(self, 'library_type_filter') and self.library_type_filter.value:
                card_type_filter = self.library_type_filter.value
            if hasattr(self, 'library_fav_filter') and self.library_fav_filter.value:
                fav_filter = self.library_fav_filter.value
            if hasattr(self, 'library_exported_filter') and self.library_exported_filter.value:
                exported_filter = self.library_exported_filter.value
            if hasattr(self, 'library_sort') and self.library_sort.value:
                sort_by = self.library_sort.value
        except:
            pass
        
        # Apply filters
        filtered_cards = []
        for card in all_cards:
            # Type filter
            if card_type_filter != "all" and card.card_type != card_type_filter:
                continue
            
            # Favorites filter
            if fav_filter == "favorited" and not card.favorited:
                continue
            if fav_filter == "not_favorited" and card.favorited:
                continue
            
            # Exported filter
            exported = getattr(card, 'exported', False)
            if exported_filter == "exported" and not exported:
                continue
            if exported_filter == "not_exported" and exported:
                continue
            
            # Search filter
            if search_text and search_text not in card.text.lower():
                continue
            
            filtered_cards.append(card)
        
        # Apply sorting
        if sort_by == "newest":
            filtered_cards.sort(key=lambda c: c.created_at or "", reverse=True)
        elif sort_by == "oldest":
            filtered_cards.sort(key=lambda c: c.created_at or "")
        elif sort_by == "alpha_asc":
            filtered_cards.sort(key=lambda c: c.text.lower())
        elif sort_by == "alpha_desc":
            filtered_cards.sort(key=lambda c: c.text.lower(), reverse=True)
        elif sort_by == "fav_first":
            filtered_cards.sort(key=lambda c: (not c.favorited, c.created_at or ""), reverse=False)
        elif sort_by == "fav_last":
            filtered_cards.sort(key=lambda c: (c.favorited, c.created_at or ""), reverse=False)
        
        self.library_filtered_cards = filtered_cards
        if reset_page:
            self.library_current_page = 0
        # Ensure current page is within valid range
        self.library_total_pages = max(1, (len(filtered_cards) + self.library_page_size - 1) // self.library_page_size)
        if self.library_current_page >= self.library_total_pages:
            self.library_current_page = max(0, self.library_total_pages - 1)
        
        logger.info(f"Applied filters: {len(filtered_cards)} cards match")
        self._display_library_page()
    
    def _display_library_page(self):
        """Display the current page of filtered cards."""
        self.library_list.controls.clear()
        # Don't clear selections here - they should persist across page changes
        
        if not self.library_filtered_cards:
            self.library_list.controls.append(
                Text("No cards match your filters. Try adjusting them!", color=Colors.GREY_500, italic=True)
            )
            self.library_stats.value = "0 cards"
            self._update_pagination_ui()
            if self.page:
                self.page.update()
            return
        
        # Calculate page range
        start = self.library_current_page * self.library_page_size
        end = start + self.library_page_size
        page_cards = self.library_filtered_cards[start:end]
        
        # Display cards
        black_count = 0
        white_count = 0
        for card in page_cards:
            is_black = card.card_type == "black"
            if is_black:
                black_count += 1
            else:
                white_count += 1
            
            emoji = "⬛" if is_black else "⬜"
            is_exported = getattr(card, 'exported', False)
            
            # Gray out exported cards
            if is_exported:
                bg_color = Colors.GREY_700 if is_black else Colors.GREY_600
                text_color = Colors.GREY_500
            else:
                bg_color = Colors.GREY_900 if is_black else Colors.GREY_800
                text_color = None  # Default
            
            # Card display with selection and delete - check if already selected
            is_selected = card.id in self.selected_library_cards
            select_cb = Checkbox(
                value=is_selected,
                on_change=lambda e, cid=card.id: self._toggle_library_selection(cid, e.control.value),
            )
            
            # Exported checkbox
            exported_cb = Checkbox(
                value=is_exported,
                label="",
                tooltip="Mark as exported to deck" if not is_exported else "Unmark as exported",
                on_change=lambda e, cid=card.id: self._toggle_card_exported(cid),
            )
            
            # Favorite toggle button
            fav_icon = Icons.STAR if card.favorited else Icons.STAR_BORDER
            fav_color = Colors.AMBER_400 if card.favorited else Colors.GREY_500
            fav_tooltip = "Remove from favorites" if card.favorited else "Add to favorites"
            
            # Format created date
            created_date = ""
            if card.created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(card.created_at.replace('Z', '+00:00'))
                    created_date = dt.strftime("%m/%d/%y")
                except:
                    created_date = card.created_at[:10] if len(card.created_at) >= 10 else ""
            
            # Exported indicator
            exported_indicator = "✅" if is_exported else ""
            
            card_row = Container(
                content=Row([
                    select_cb,
                    Text(emoji, size=16),
                    Text(card.text, size=12, expand=True, selectable=True, color=text_color),
                    Text(exported_indicator, size=12, tooltip="Exported to deck"),
                    exported_cb,
                    Text(created_date, size=10, color=Colors.GREY_500, width=60),
                    IconButton(
                        icon=fav_icon,
                        icon_size=14,
                        icon_color=fav_color,
                        tooltip=fav_tooltip,
                        on_click=lambda e, cid=card.id: self._toggle_card_favorite(cid),
                    ),
                    IconButton(
                        icon=Icons.EDIT,
                        icon_size=14,
                        tooltip="Edit card",
                        on_click=lambda e, cid=card.id, txt=card.text: self._edit_card_dialog(cid, txt),
                    ),
                    IconButton(
                        icon=Icons.CONTENT_COPY,
                        icon_size=14,
                        tooltip="Copy",
                        on_click=lambda e, t=card.text: self._copy_to_clipboard(t),
                    ),
                    IconButton(
                        icon=Icons.DELETE_OUTLINE,
                        icon_size=14,
                        icon_color=Colors.RED_400,
                        tooltip="Delete",
                        on_click=lambda e, cid=card.id: self._delete_library_card(cid),
                    ),
                ], spacing=8),
                bgcolor=bg_color,
                padding=padding.all(8),
                border_radius=border_radius.all(6),
            )
            self.library_list.controls.append(card_row)
        
        # Update stats
        total_filtered = len(self.library_filtered_cards)
        total_all = len(self.cah_generator.get_cards())
        total_black = len([c for c in self.library_filtered_cards if c.card_type == "black"])
        total_white = len([c for c in self.library_filtered_cards if c.card_type == "white"])
        total_fav = len([c for c in self.library_filtered_cards if c.favorited])
        
        filter_text = f" ({total_filtered}/{total_all} filtered)" if total_filtered < total_all else ""
        self.library_stats.value = f"📊 {total_filtered} cards ({total_black} ⬛, {total_white} ⬜, {total_fav} ⭐){filter_text}"
        
        logger.info(f"Displayed {len(page_cards)} cards on page {self.library_current_page + 1}, library_list has {len(self.library_list.controls)} controls")
        self._update_pagination_ui()
        self._update_library_selection_ui()
        if self.page:
            self.page.update()
    
    def _update_pagination_ui(self):
        """Update pagination button states and labels."""
        self.library_page_label.value = f"Page {self.library_current_page + 1} of {self.library_total_pages}"
        self.library_prev_btn.disabled = self.library_current_page == 0
        self.library_next_btn.disabled = self.library_current_page >= self.library_total_pages - 1
        try:
            self.page.update()
        except:
            pass
    
    def _library_prev_page(self, e=None):
        """Go to previous page."""
        if self.library_current_page > 0:
            self.library_current_page -= 1
            self._display_library_page()
    
    def _library_next_page(self, e=None):
        """Go to next page."""
        if self.library_current_page < self.library_total_pages - 1:
            self.library_current_page += 1
            self._display_library_page()
    
    def _toggle_library_selection(self, card_id: str, selected: bool):
        """Toggle selection of a library card."""
        if selected:
            self.selected_library_cards.add(card_id)
        else:
            self.selected_library_cards.discard(card_id)
        self._update_library_selection_ui()
    
    def _select_page(self, e=None):
        """Select only cards on current page (clears other selections first)."""
        # Clear all selections first
        self.selected_library_cards.clear()
        
        # Get current page cards and select them
        start = self.library_current_page * self.library_page_size
        end = start + self.library_page_size
        page_cards = self.library_filtered_cards[start:end]
        
        for card in page_cards:
            self.selected_library_cards.add(card.id)
        
        self._display_library_page()
        self._update_library_selection_ui()
    
    def _select_all_filtered(self, e=None):
        """Select all cards in current filtered view."""
        for card in self.library_filtered_cards:
            self.selected_library_cards.add(card.id)
        
        self._display_library_page()
        self._update_library_selection_ui()
    
    def _clear_selection(self, e=None):
        """Clear all selections."""
        self.selected_library_cards.clear()
        self._display_library_page()
        self._update_library_selection_ui()
    
    def _edit_card_dialog(self, card_id: str, current_text: str):
        """Open a dialog to edit card text."""
        edit_field = TextField(
            value=current_text,
            multiline=True,
            min_lines=2,
            max_lines=5,
            expand=True,
        )
        
        def save_edit(e):
            new_text = edit_field.value.strip()
            if new_text and self.cah_generator:
                if self.cah_generator.update_card_text(card_id, new_text):
                    self.page.close(dlg)
                    self._apply_library_filters()
                    self.page.snack_bar = SnackBar(content=Text("✏️ Card updated"))
                    self.page.snack_bar.open = True
                    self.page.update()
        
        def cancel_edit(e):
            self.page.close(dlg)
        
        dlg = AlertDialog(
            modal=True,
            title=Text("✏️ Edit Card"),
            content=Container(
                content=edit_field,
                width=400,
            ),
            actions=[
                TextButton("Cancel", on_click=cancel_edit),
                ElevatedButton("Save", on_click=save_edit),
            ],
            actions_alignment=MainAxisAlignment.END,
        )
        
        self.page.open(dlg)
    
    def _delete_library_card(self, card_id: str):
        """Delete a single card from the library."""
        if self.cah_generator:
            self.cah_generator.delete_card(card_id)
            self.selected_library_cards.discard(card_id)
            # Re-apply filters without resetting page
            self._apply_library_filters(reset_page=False)
            self.page.snack_bar = SnackBar(content=Text("🗑️ Card deleted"))
            self.page.snack_bar.open = True
            self.page.update()
    
    def _delete_selected_library(self, e=None):
        """Delete all selected library cards."""
        if not self.selected_library_cards or not self.cah_generator:
            return
        
        count = len(self.selected_library_cards)
        for card_id in list(self.selected_library_cards):
            self.cah_generator.delete_card(card_id)
        
        self.selected_library_cards.clear()
        # Re-apply filters without resetting page
        self._apply_library_filters(reset_page=False)
        self.page.snack_bar = SnackBar(content=Text(f"🗑️ Deleted {count} cards"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _toggle_card_exported(self, card_id: str):
        """Toggle exported status for a single card."""
        if self.cah_generator:
            self.cah_generator.toggle_exported(card_id)
            self._apply_library_filters(reset_page=False)
    
    def _export_selected_library(self, e=None):
        """Mark all selected cards as exported."""
        if not self.cah_generator:
            return
        count = 0
        for card_id in self.selected_library_cards:
            for card in self.cah_generator.cards:
                if card.id == card_id and not getattr(card, 'exported', False):
                    card.exported = True
                    count += 1
                    break
        if count > 0:
            self.cah_generator._save_cards()
        self.selected_library_cards.clear()
        self._apply_library_filters(reset_page=False)
        self.page.snack_bar = SnackBar(content=Text(f"✅ Marked {count} cards as exported"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _unexport_selected_library(self, e=None):
        """Unmark all selected cards as exported."""
        if not self.cah_generator:
            return
        count = 0
        for card_id in self.selected_library_cards:
            for card in self.cah_generator.cards:
                if card.id == card_id and getattr(card, 'exported', False):
                    card.exported = False
                    count += 1
                    break
        if count > 0:
            self.cah_generator._save_cards()
        self.selected_library_cards.clear()
        self._apply_library_filters(reset_page=False)
        self.page.snack_bar = SnackBar(content=Text(f"⬜ Unmarked {count} cards as exported"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _update_library_selection_ui(self):
        """Update the library selection count and delete/favorite/export buttons."""
        count = len(self.selected_library_cards)
        if hasattr(self, 'delete_selected_lib_btn'):
            self.delete_selected_lib_btn.visible = count > 0
            self.delete_selected_lib_btn.text = f"🗑️ Delete Selected ({count})"
        if hasattr(self, 'favorite_selected_lib_btn'):
            self.favorite_selected_lib_btn.visible = count > 0
            self.favorite_selected_lib_btn.text = f"⭐ Favorite Selected ({count})"
        if hasattr(self, 'unfavorite_selected_lib_btn'):
            self.unfavorite_selected_lib_btn.visible = count > 0
            self.unfavorite_selected_lib_btn.text = f"☆ Unfavorite Selected ({count})"
        if hasattr(self, 'export_selected_lib_btn'):
            self.export_selected_lib_btn.visible = count > 0
            self.export_selected_lib_btn.text = f"✅ Mark Exported ({count})"
        if hasattr(self, 'unexport_selected_lib_btn'):
            self.unexport_selected_lib_btn.visible = count > 0
            self.unexport_selected_lib_btn.text = f"⬜ Unmark Exported ({count})"
        if hasattr(self, 'clear_selection_lib_btn'):
            self.clear_selection_lib_btn.visible = count > 0
        if hasattr(self, 'lib_selection_count'):
            self.lib_selection_count.value = f"{count} selected" if count > 0 else ""
        try:
            self.page.update()
        except:
            pass
    
    def _toggle_card_favorite(self, card_id: str):
        """Toggle favorite status for a single card."""
        if self.cah_generator:
            self.cah_generator.toggle_favorite(card_id)
            # Refresh display but keep filters and page
            self._apply_library_filters(reset_page=False)
    
    def _favorite_selected_library(self, e=None):
        """Mark all selected cards as favorites."""
        if not self.cah_generator:
            return
        count = 0
        for card_id in self.selected_library_cards:
            # Find card and set favorite if not already
            for card in self.cah_generator.cards:
                if card.id == card_id and not card.favorited:
                    card.favorited = True
                    count += 1
                    break
        if count > 0:
            self.cah_generator._save_cards()
        self.selected_library_cards.clear()
        self._apply_library_filters(reset_page=False)
        self.page.snack_bar = SnackBar(content=Text(f"⭐ Added {count} cards to favorites"))
        self.page.snack_bar.open = True
        self.page.update()
    
    def _unfavorite_selected_library(self, e=None):
        """Remove all selected cards from favorites."""
        if not self.cah_generator:
            return
        count = 0
        for card_id in self.selected_library_cards:
            # Find card and remove favorite if set
            for card in self.cah_generator.cards:
                if card.id == card_id and card.favorited:
                    card.favorited = False
                    count += 1
                    break
        if count > 0:
            self.cah_generator._save_cards()
        self.selected_library_cards.clear()
        self._apply_library_filters(reset_page=False)
        self.page.snack_bar = SnackBar(content=Text(f"☆ Removed {count} cards from favorites"))
        self.page.snack_bar.open = True
        self.page.update()

    def _toggle_favorite(self, card: CAHCard):
        """Toggle favorite status for a card."""
        if self.cah_generator:
            self.cah_generator.toggle_favorite(card.id)
            self._load_library()
    
    def _export_cards(self, format: str):
        """Export cards to file - opens file save dialog."""
        logger.info(f"_export_cards called with format={format}")
        if not self.cah_generator:
            logger.warning("No cah_generator for export")
            return
        
        cards = self.cah_generator.get_cards()
        logger.info(f"Exporting {len(cards)} cards")
        if not cards:
            self.page.snack_bar = SnackBar(content=Text("❌ No cards to export"))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Store format for the callback
        self._pending_export_format = format
        
        # Open save file dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"cards_{timestamp}.{format}"
        
        self.file_picker.save_file(
            dialog_title="Save Cards Export",
            file_name=default_name,
            allowed_extensions=[format],
        )
    
    def _on_export_file_picked(self, e: ft.FilePickerResultEvent):
        """Handle file save dialog result."""
        if not e.path or not self._pending_export_format:
            return
        
        format = self._pending_export_format
        filepath = Path(e.path)
        
        try:
            cards = self.cah_generator.get_cards()
            
            if format == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump([{
                        "text": c.text,
                        "type": c.card_type,
                        "style": c.style,
                        "topic": c.topic,
                        "favorited": c.favorited,
                    } for c in cards], f, indent=2)
            else:  # txt
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("=== BLACK CARDS (Prompts) ===\n\n")
                    for c in cards:
                        if c.card_type == "black":
                            f.write(f"⬛ {c.text}\n")
                    f.write("\n\n=== WHITE CARDS (Answers) ===\n\n")
                    for c in cards:
                        if c.card_type == "white":
                            f.write(f"⬜ {c.text}\n")
            
            # Show success dialog
            def close_dialog(e):
                dialog.open = False
                self.page.update()
            
            dialog = AlertDialog(
                title=Text("✅ Export Successful"),
                content=Text(f"Exported {len(cards)} cards to:\n{filepath}"),
                actions=[
                    TextButton("OK", on_click=close_dialog),
                ],
            )
            self.page.overlay.append(dialog)
            dialog.open = True
            self.page.update()
            
            logger.info(f"Exported {len(cards)} cards to {filepath}")
            
        except Exception as ex:
            logger.error(f"Export error: {ex}")
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Export failed: {str(ex)}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
        
        finally:
            self._pending_export_format = None
    
    def _delete_last_n(self, e):
        """Delete the last N cards."""
        if not self.cah_generator:
            return
        
        n = int(self.delete_n_slider.value)
        cards = self.cah_generator.get_cards()
        
        if len(cards) == 0:
            return
        
        # Remove last N cards
        remaining = cards[:-n] if n < len(cards) else []
        self.cah_generator.cards = remaining
        self.cah_generator._save_cards()
        
        self.page.snack_bar = SnackBar(content=Text(f"🗑️ Deleted {min(n, len(cards))} cards"))
        self.page.snack_bar.open = True
        self._load_library()
    
    def _confirm_clear_all(self, e):
        """Confirm before clearing all cards."""
        if not self.cah_generator:
            return
        
        cards = self.cah_generator.get_cards()
        if not cards:
            return
        
        def do_clear(e):
            self.cah_generator.cards = []
            self.cah_generator._save_cards()
            dialog.open = False
            self.page.snack_bar = SnackBar(content=Text("🗑️ All cards cleared"))
            self.page.snack_bar.open = True
            self._load_library()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = AlertDialog(
            title=Text("Clear All Cards?"),
            content=Text(f"Are you sure you want to delete all {len(cards)} cards?\nThis cannot be undone."),
            actions=[
                TextButton("Cancel", on_click=cancel),
                TextButton("Clear All", on_click=do_clear),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _build_generate_tab(self) -> Container:
        """Build the Generate sub-tab content."""
        settings_panel = Container(
            content=Column([
                Text("⚙️ Settings", size=16, weight=FontWeight.BOLD),
                Divider(height=10, color=Colors.TRANSPARENT),
                
                self.topic_input,
                
                Row([
                    self.card_type_dropdown,
                    self.style_dropdown,
                ], spacing=15, wrap=True),
                
                self.custom_style_input,
                
                # LoRA selection row
                Row([
                    self.lora_dropdown,
                    self.lora_refresh_btn,
                ], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                
                Row([
                    Text("📊 Quantity:", size=12),
                    self.quantity_value,
                    self.quantity_slider,
                ], spacing=10),
                
                self.generate_btn,
                self.progress_bar,
                self.status_text,
                
                Divider(height=20, color=Colors.TRANSPARENT),
                
                Container(
                    content=Column([
                        Text("💡 Tips:", size=12, weight=FontWeight.BOLD, color=Colors.BLUE_400),
                        Text("• Load a chat model first", size=11, color=Colors.GREY_400),
                        Text("• Be specific with your topic", size=11, color=Colors.GREY_400),
                        Text("• Select a trained LoRA for style", size=11, color=Colors.GREY_400),
                        Text("• Classic style works best for humor", size=11, color=Colors.GREY_400),
                    ], spacing=3),
                    padding=padding.all(10),
                    bgcolor=Colors.with_opacity(0.1, Colors.BLUE),
                    border_radius=border_radius.all(8),
                ),
            ], spacing=10),
            padding=padding.all(15),
            width=350,
        )
        
        output_panel = Container(
            content=Column([
                Row([
                    Text("🎴 Generated Cards", size=16, weight=FontWeight.BOLD),
                    Container(expand=True),
                    self.gen_selection_count,
                    self.delete_selected_gen_btn,
                ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                Divider(height=10, color=Colors.TRANSPARENT),
                self.cards_output,
            ], spacing=5, expand=True),
            expand=True,
            padding=padding.all(15),
        )
        
        return Container(
            content=Row([
                settings_panel,
                ft.VerticalDivider(width=1, color=Colors.GREY_800),
                output_panel,
            ], spacing=0, expand=True),
            expand=True,
        )
    
    def _build_article_tab(self) -> Container:
        """Build the From Article sub-tab content."""
        settings_panel = Container(
            content=Column([
                Text("⚙️ Article Settings", size=16, weight=FontWeight.BOLD),
                Divider(height=10, color=Colors.TRANSPARENT),
                
                self.article_input,
                
                self.article_card_type,
                
                Row([
                    Text("📊 Cards to Extract:", size=12),
                    self.article_quantity_value,
                    self.article_quantity_slider,
                ], spacing=10),
                
                self.extract_btn,
                self.article_progress,
                self.article_status,
                
                Divider(height=20, color=Colors.TRANSPARENT),
                
                Container(
                    content=Column([
                        Text("💡 How it works:", size=12, weight=FontWeight.BOLD, color=Colors.BLUE_400),
                        Text("• Paste article text above", size=11, color=Colors.GREY_400),
                        Text("• AI finds the funniest lines", size=11, color=Colors.GREY_400),
                        Text("• Adjusts verb tenses for CAH format", size=11, color=Colors.GREY_400),
                        Text("• Works great with Clickhole, The Onion", size=11, color=Colors.GREY_400),
                    ], spacing=3),
                    padding=padding.all(10),
                    bgcolor=Colors.with_opacity(0.1, Colors.BLUE),
                    border_radius=border_radius.all(8),
                ),
            ], spacing=10, scroll=ScrollMode.AUTO),
            padding=padding.all(15),
            width=400,
        )
        
        output_panel = Container(
            content=Column([
                Text("🎴 Extracted Cards", size=16, weight=FontWeight.BOLD),
                Divider(height=10, color=Colors.TRANSPARENT),
                self.extracted_cards_output,
            ], spacing=5),
            expand=True,
            padding=padding.all(15),
        )
        
        return Container(
            content=Row([
                settings_panel,
                ft.VerticalDivider(width=1, color=Colors.GREY_800),
                output_panel,
            ], spacing=0, expand=True),
            expand=True,
        )
    
    def _build_library_tab(self) -> Container:
        """Build the My Cards (Library) sub-tab content."""
        
        # Load cards
        self._load_library()
        
        # SIMPLE FLAT STRUCTURE - no nested containers
        return Column([
            # Header row
            Row([
                Text("📚 Saved Cards", size=16, weight=FontWeight.BOLD),
                Container(expand=True),
                self.library_stats,
                self.refresh_library_btn,
            ]),
            # Filter row
            Row([
                self.library_search_field,
                self.library_type_filter,
                self.library_fav_filter,
                self.library_exported_filter,
                self.library_sort,
            ], spacing=10),
            # Action buttons row
            Row([
                self.export_json_btn,
                self.export_txt_btn,
                Container(width=10),
                self.select_page_lib_btn,
                self.select_all_lib_btn,
                self.clear_selection_lib_btn,
                self.lib_selection_count,
            ], spacing=10),
            # Bulk action buttons row (only visible when cards selected)
            Row([
                self.favorite_selected_lib_btn,
                self.unfavorite_selected_lib_btn,
                self.export_selected_lib_btn,
                self.unexport_selected_lib_btn,
                self.delete_selected_lib_btn,
            ], spacing=10),
            # Pagination row
            Row([
                self.library_prev_btn,
                self.library_page_label,
                self.library_next_btn,
            ], spacing=5),
            # Cards list - directly in Column, no wrapper
            self.library_list,
        ], spacing=10, expand=True)
    
    def build(self) -> Container:
        """Build the Card Generator tab with sub-tabs."""
        
        # Update model status
        self._check_model_loaded()
        
        # Load LoRA list (deferred from init to avoid issues)
        try:
            self._load_lora_list()
        except Exception as ex:
            logger.error(f"Failed to load LoRA list on build: {ex}")
        
        # Create sub-tabs
        sub_tabs = Tabs(
            selected_index=0,
            animation_duration=200,
            tabs=[
                Tab(
                    text="🎴 Generate",
                    icon=Icons.AUTO_AWESOME,
                    content=self._build_generate_tab(),
                ),
                Tab(
                    text="📰 From Article",
                    icon=Icons.ARTICLE,
                    content=self._build_article_tab(),
                ),
                Tab(
                    text="📚 My Cards",
                    icon=Icons.COLLECTIONS_BOOKMARK,
                    content=self._build_library_tab(),
                ),
            ],
            expand=True,
        )
        
        return Container(
            content=Column([
                # Header with no-model banner
                Container(
                    content=Column([
                        Row([
                            Text("🃏 Card Generator", size=20, weight=FontWeight.BOLD),
                            Text("Generate custom party game cards using the text model", 
                                 size=12, color=Colors.GREY_400),
                        ], spacing=15),
                        self.no_model_banner,
                    ], spacing=10),
                    padding=padding.only(left=15, top=10, right=15, bottom=5),
                ),
                # Sub-tabs
                sub_tabs,
            ], spacing=0),
            expand=True,
        )
