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
        
        self.library_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
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
    
    def _add_card_to_output(self, card_text: str, card_type: str, output_list: ListView):
        """Add a card item to the output list."""
        is_black = card_type == "black"
        emoji = "⬛" if is_black else "⬜"
        bg_color = Colors.GREY_900 if is_black else Colors.GREY_800
        
        card_item = Container(
            content=Row([
                Text(emoji, size=20),
                Text(card_text, size=13, expand=True, selectable=True),
                IconButton(
                    icon=Icons.CONTENT_COPY,
                    icon_size=16,
                    tooltip="Copy",
                    on_click=lambda e, t=card_text: self._copy_to_clipboard(t),
                ),
            ], spacing=10),
            bgcolor=bg_color,
            padding=padding.all(12),
            border_radius=border_radius.all(8),
        )
        output_list.controls.append(card_item)
    
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
            try:
                from core.chat_gen import generate_chat_response
                
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
                
                # Generate using the chat model
                success, response = generate_chat_response(
                    user_message=user_prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=800,
                    temperature=0.8,
                )
                
                if not success:
                    raise Exception(response)
                
                # Parse the response
                cards = self._parse_card_response(response, topic, style_enum, type_enum)
                
                # Update UI on main thread
                def update_ui():
                    self.cards_output.controls.clear()
                    
                    if cards:
                        for card in cards:
                            self._add_card_to_output(card["text"], card["type"], self.cards_output)
                        
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
        except json.JSONDecodeError:
            pass
        
        # Fallback: line-by-line parsing
        if not cards:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("```"):
                    continue
                
                # Remove list markers
                if line[0] in "-*•0123456789":
                    line = line.lstrip("-*•0123456789.) ").strip()
                
                # Remove emoji prefixes
                if line.startswith(("⬛", "⬜")):
                    line = line[1:].strip()
                
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
        """Load saved cards into the library view."""
        self.library_list.controls.clear()
        
        if not self.cah_generator:
            self.library_list.controls.append(
                Text("⚠️ Card generator not available", color=Colors.GREY_500)
            )
            self.library_stats.value = "No cards loaded"
            self.page.update()
            return
        
        cards = self.cah_generator.get_cards()
        
        if not cards:
            self.library_list.controls.append(
                Container(
                    content=Column([
                        ft.Icon(Icons.INBOX, size=48, color=Colors.GREY_600),
                        Text("No saved cards yet", size=14, color=Colors.GREY_500),
                        Text("Generate some cards to see them here!", size=12, color=Colors.GREY_600),
                    ], horizontal_alignment=CrossAxisAlignment.CENTER, spacing=10),
                    alignment=alignment.center,
                    padding=padding.all(30),
                )
            )
            self.library_stats.value = "0 cards"
        else:
            black_count = 0
            white_count = 0
            
            for card in cards:
                is_black = card.card_type == "black"
                if is_black:
                    black_count += 1
                else:
                    white_count += 1
                
                emoji = "⬛" if is_black else "⬜"
                bg_color = Colors.GREY_900 if is_black else Colors.GREY_800
                
                card_item = Container(
                    content=Row([
                        Text(emoji, size=18),
                        Column([
                            Text(card.text, size=12, selectable=True),
                            Text(f"📌 {card.topic} | 🎨 {card.style}", 
                                 size=10, color=Colors.GREY_500),
                        ], spacing=3, expand=True),
                        Row([
                            IconButton(
                                icon=Icons.CONTENT_COPY,
                                icon_size=16,
                                tooltip="Copy",
                                on_click=lambda e, t=card.text: self._copy_to_clipboard(t),
                            ),
                            IconButton(
                                icon=Icons.STAR_BORDER if not card.favorited else Icons.STAR,
                                icon_size=16,
                                icon_color=Colors.YELLOW_400 if card.favorited else None,
                                tooltip="Favorite",
                                on_click=lambda e, c=card: self._toggle_favorite(c),
                            ),
                        ], spacing=0),
                    ], spacing=10),
                    bgcolor=bg_color,
                    padding=padding.all(10),
                    border_radius=border_radius.all(8),
                )
                self.library_list.controls.append(card_item)
            
            self.library_stats.value = f"📊 {len(cards)} cards ({black_count} ⬛ black, {white_count} ⬜ white)"
        
        self.page.update()
    
    def _toggle_favorite(self, card: CAHCard):
        """Toggle favorite status for a card."""
        if self.cah_generator:
            self.cah_generator.toggle_favorite(card.id)
            self._load_library()
    
    def _export_cards(self, format: str):
        """Export cards to file."""
        if not self.cah_generator:
            return
        
        cards = self.cah_generator.get_cards()
        if not cards:
            self.page.snack_bar = SnackBar(content=Text("No cards to export"))
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        try:
            export_dir = Path("data/cah/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                filepath = export_dir / f"cards_{timestamp}.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump([{
                        "text": c.text,
                        "type": c.card_type,
                        "style": c.style,
                        "topic": c.topic,
                        "favorited": c.favorited,
                    } for c in cards], f, indent=2)
            else:  # txt
                filepath = export_dir / f"cards_{timestamp}.txt"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("=== BLACK CARDS (Prompts) ===\n\n")
                    for c in cards:
                        if c.card_type == "black":
                            f.write(f"⬛ {c.text}\n")
                    f.write("\n\n=== WHITE CARDS (Answers) ===\n\n")
                    for c in cards:
                        if c.card_type == "white":
                            f.write(f"⬜ {c.text}\n")
            
            self.page.snack_bar = SnackBar(
                content=Text(f"✅ Exported {len(cards)} cards to {filepath.name}")
            )
            self.page.snack_bar.open = True
            self.page.update()
            
        except Exception as ex:
            logger.error(f"Export error: {ex}")
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ Export failed: {str(ex)}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
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
                        Text("• Classic style works best for humor", size=11, color=Colors.GREY_400),
                        Text("• Try Nerdy for gaming/tech references", size=11, color=Colors.GREY_400),
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
                Text("🎴 Generated Cards", size=16, weight=FontWeight.BOLD),
                Divider(height=10, color=Colors.TRANSPARENT),
                self.cards_output,
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
        # Load library on tab build
        self._load_library()
        
        header = Row([
            Text("📚 Saved Cards", size=16, weight=FontWeight.BOLD),
            Container(expand=True),
            self.library_stats,
            self.refresh_library_btn,
        ], alignment=MainAxisAlignment.SPACE_BETWEEN)
        
        actions = Row([
            self.export_json_btn,
            self.export_txt_btn,
            Container(expand=True),
            Row([
                Text("Delete last:", size=11),
                self.delete_n_value,
                self.delete_n_slider,
                self.delete_last_btn,
            ], spacing=5),
            self.clear_all_btn,
        ], spacing=10, wrap=True)
        
        return Container(
            content=Column([
                header,
                actions,
                Divider(height=10, color=Colors.TRANSPARENT),
                self.library_list,
            ], spacing=10),
            padding=padding.all(15),
            expand=True,
        )
    
    def build(self) -> Container:
        """Build the Card Generator tab with sub-tabs."""
        
        # Update model status
        self._check_model_loaded()
        
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
