"""
Card LoRA Training Tab for KVGenius Flet App
Provides UI for training card generation LoRAs (CAH-style card themes).
Extends the base training UI with card-specific data input.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField,
    ElevatedButton, IconButton, Card, Dropdown, FilePicker,
    Icons, Colors, FontWeight, ScrollMode, Tabs, Tab,
    padding, border_radius, alignment,
    CrossAxisAlignment, MainAxisAlignment,
    dropdown, FilePickerResultEvent
)

from src.training.text_lora_trainer import (
    TextLoRAType, TextTrainingExample, create_card_generation_examples
)
from ui.tabs.base_text_training import BaseTextTrainingTab

logger = logging.getLogger(__name__)


class CardLoRATrainingTab(BaseTextTrainingTab):
    """
    Card LoRA Training Tab - Train LoRAs for card generation.
    Allows training custom CAH-style card themes and humor styles.
    
    Extends BaseTextTrainingTab with card-specific data input UI.
    """
    
    def get_lora_type(self) -> TextLoRAType:
        return TextLoRAType.CARD_GENERATOR
    
    def get_tab_title(self) -> str:
        return "Train Card LoRA"
    
    def get_tab_icon(self) -> str:
        return "🃏"
    
    def _build_common_ui(self):
        """Build UI components - extends parent with card-specific fields."""
        super()._build_common_ui()
        
        # === Theme Configuration ===
        self.theme_name_field = TextField(
            label="Theme Name",
            hint_text="e.g., Star Wars, Tech Humor, Dad Jokes",
            width=300,
            on_change=self._validate_form,
        )
        
        self.theme_description_field = TextField(
            label="Theme Description",
            hint_text="Brief description of the card theme and humor style",
            multiline=True,
            min_lines=2,
            max_lines=3,
            expand=True,
        )
        
        # === Card Type Tabs ===
        self.black_cards_area = TextField(
            label="Black Cards (one per line)",
            hint_text="What's the next big thing in ___?\nI never thought ___ would be so ___.",
            multiline=True,
            min_lines=10,
            max_lines=20,
            expand=True,
            on_change=self._on_cards_change,
        )
        
        self.white_cards_area = TextField(
            label="White Cards (one per line)",
            hint_text="A disappointing birthday party\nEmbarrassing childhood photos\nCrying in the shower",
            multiline=True,
            min_lines=10,
            max_lines=20,
            expand=True,
            on_change=self._on_cards_change,
        )
        
        # File picker for bulk import
        self.file_picker = FilePicker(
            on_result=self._on_file_picked,
        )
        
        self.file_picker_btn = ElevatedButton(
            "📁 Import from File",
            icon=Icons.UPLOAD_FILE,
            on_click=lambda e: self.file_picker.pick_files(
                allowed_extensions=["json", "txt", "csv"]
            ),
        )
        
        self.file_status = Text("", size=11, color=Colors.GREY_400)
        
        # Load from saved cards button
        self.load_saved_btn = ElevatedButton(
            "📚 Load from Card Library",
            icon=Icons.LIBRARY_BOOKS,
            on_click=self._load_from_library,
        )
        
        # Card counts
        self.black_count_text = Text("0 black cards", size=11, color=Colors.GREY_500)
        self.white_count_text = Text("0 white cards", size=11, color=Colors.GREY_500)
        self.total_examples_text = Text("0 training examples", size=12, color=Colors.GREY_500)
    
    def _on_cards_change(self, e=None):
        """Handle card text area changes."""
        self._update_card_counts()
        self._validate_form()
    
    def _update_card_counts(self):
        """Update the card count displays."""
        black_cards = self._parse_cards(self.black_cards_area.value or "")
        white_cards = self._parse_cards(self.white_cards_area.value or "")
        
        self.black_count_text.value = f"{len(black_cards)} black cards"
        self.white_count_text.value = f"{len(white_cards)} white cards"
        
        total = len(black_cards) + len(white_cards)
        self.total_examples_text.value = f"{total} training examples"
        
        if total >= 3:
            self.total_examples_text.color = Colors.GREEN_400
        elif total > 0:
            self.total_examples_text.color = Colors.YELLOW_400
        else:
            self.total_examples_text.color = Colors.GREY_500
        
        # Update individual counts colors
        self.black_count_text.color = Colors.GREEN_400 if len(black_cards) >= 5 else (Colors.YELLOW_400 if len(black_cards) > 0 else Colors.GREY_500)
        self.white_count_text.color = Colors.GREEN_400 if len(white_cards) >= 5 else (Colors.YELLOW_400 if len(white_cards) > 0 else Colors.GREY_500)
    
    def _parse_cards(self, text: str) -> List[str]:
        """Parse cards from text (one per line)."""
        cards = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty and comment lines
                cards.append(line)
        return cards
    
    def _on_file_picked(self, e: FilePickerResultEvent):
        """Handle file picker result."""
        if not e.files:
            return
        
        file_path = e.files[0].path
        self.file_status.value = f"Loading: {Path(file_path).name}..."
        self.page.update()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.endswith('.csv'):
                import csv
                import io
                black_cards = []
                white_cards = []
                
                # Parse CSV - handle both standard format and PROMPT/RESPONSE format
                lines = content.strip().split('\n')
                
                # Check if first line looks like headers
                first_line_parts = lines[0].split(',', 1) if lines else []
                is_header_format = len(first_line_parts) == 2 and first_line_parts[0].upper() in ['PROMPT', 'RESPONSE', 'BLACK', 'WHITE']
                is_kv_format = len(first_line_parts) == 2 and first_line_parts[0].upper() in ['PROMPT', 'RESPONSE']
                
                if is_kv_format and not (first_line_parts[0].upper() == 'PROMPT' and first_line_parts[1].upper() == 'RESPONSE'):
                    # Key-value format: each line is PROMPT,<text> or RESPONSE,<text>
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().upper()
                            value = parts[1].strip().strip('"')  # Remove quotes if present
                            
                            if key == 'PROMPT':
                                black_cards.append(value)
                            elif key == 'RESPONSE':
                                white_cards.append(value)
                else:
                    # Standard CSV format with headers
                    csv_reader = csv.DictReader(io.StringIO(content))
                    for row in csv_reader:
                        # Look for common column name patterns
                        prompt = row.get('PROMPT') or row.get('prompt') or row.get('black') or row.get('Black')
                        response = row.get('RESPONSE') or row.get('response') or row.get('white') or row.get('White')
                        
                        # Try to determine card type
                        if prompt:
                            # If it has blanks (___), it's a black card
                            if '___' in prompt or prompt.endswith('?'):
                                black_cards.append(prompt)
                            else:
                                white_cards.append(prompt)
                        
                        if response:
                            white_cards.append(response)
                
                # Update text areas
                if black_cards:
                    existing = self.black_cards_area.value or ""
                    if existing.strip():
                        existing += "\n"
                    self.black_cards_area.value = existing + "\n".join(black_cards)
                
                if white_cards:
                    existing = self.white_cards_area.value or ""
                    if existing.strip():
                        existing += "\n"
                    self.white_cards_area.value = existing + "\n".join(white_cards)
                
                self.file_status.value = f"✅ Loaded {len(black_cards)} black + {len(white_cards)} white cards from CSV"
                self.file_status.color = Colors.GREEN_400
                
            elif file_path.endswith('.json'):
                data = json.loads(content)
                
                # Handle various JSON formats
                black_cards = []
                white_cards = []
                
                if isinstance(data, dict):
                    black_cards = data.get('black_cards', data.get('black', []))
                    white_cards = data.get('white_cards', data.get('white', []))
                elif isinstance(data, list):
                    # Assume list of cards with type field
                    for card in data:
                        if isinstance(card, dict):
                            card_type = card.get('type', card.get('card_type', '')).lower()
                            text = card.get('text', card.get('content', ''))
                            if 'black' in card_type:
                                black_cards.append(text)
                            elif 'white' in card_type:
                                white_cards.append(text)
                        elif isinstance(card, str):
                            # Can't determine type, add to white
                            white_cards.append(card)
                
                # Update text areas
                if black_cards:
                    existing = self.black_cards_area.value or ""
                    if existing.strip():
                        existing += "\n"
                    self.black_cards_area.value = existing + "\n".join(black_cards)
                
                if white_cards:
                    existing = self.white_cards_area.value or ""
                    if existing.strip():
                        existing += "\n"
                    self.white_cards_area.value = existing + "\n".join(white_cards)
                
                self.file_status.value = f"✅ Loaded {len(black_cards)} black + {len(white_cards)} white cards"
                self.file_status.color = Colors.GREEN_400
                
            else:
                # Plain text - assume one card per line, auto-detect type
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # If contains ___, it's likely a black card
                        if '___' in line or line.endswith('?'):
                            existing = self.black_cards_area.value or ""
                            if existing.strip():
                                existing += "\n"
                            self.black_cards_area.value = existing + line
                        else:
                            existing = self.white_cards_area.value or ""
                            if existing.strip():
                                existing += "\n"
                            self.white_cards_area.value = existing + line
                
                self.file_status.value = f"✅ Loaded cards from {Path(file_path).name}"
                self.file_status.color = Colors.GREEN_400
            
        except Exception as ex:
            self.file_status.value = f"❌ Error loading file: {ex}"
            self.file_status.color = Colors.RED_400
        
        self._on_cards_change()
        self.page.update()
    
    def _load_from_library(self, e):
        """Load cards from the saved card library."""
        try:
            from src.cards.cah_generator import CAHGenerator
            
            generator = CAHGenerator()
            cards = generator.get_cards()
            
            if not cards:
                self.page.snack_bar = ft.SnackBar(
                    content=Text("No saved cards found in library"),
                    bgcolor=Colors.ORANGE_700,
                )
                self.page.snack_bar.open = True
                self.page.update()
                return
            
            # Separate by type
            black_cards = []
            white_cards = []
            
            for card in cards:
                card_type = card.get('type', card.get('card_type', '')).lower()
                text = card.get('text', card.get('content', ''))
                if 'black' in card_type:
                    black_cards.append(text)
                elif 'white' in card_type:
                    white_cards.append(text)
            
            # Update text areas
            if black_cards:
                self.black_cards_area.value = "\n".join(black_cards)
            if white_cards:
                self.white_cards_area.value = "\n".join(white_cards)
            
            self._on_cards_change()
            
            self.page.snack_bar = ft.SnackBar(
                content=Text(f"✅ Loaded {len(black_cards)} black + {len(white_cards)} white cards from library"),
                bgcolor=Colors.GREEN_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            
        except Exception as ex:
            logger.error(f"Error loading from library: {ex}")
            self.page.snack_bar = ft.SnackBar(
                content=Text(f"❌ Error loading from library: {ex}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def get_training_examples(self) -> List[TextTrainingExample]:
        """Extract training examples from the UI inputs."""
        theme_name = self.theme_name_field.value or ""
        
        black_cards = self._parse_cards(self.black_cards_area.value or "")
        white_cards = self._parse_cards(self.white_cards_area.value or "")
        
        return create_card_generation_examples(
            black_cards=black_cards,
            white_cards=white_cards,
            theme_name=theme_name
        )
    
    def build_data_input_section(self) -> Container:
        """Build the data input section for card training."""
        # Add file picker to page overlay
        self.page.overlay.append(self.file_picker)
        
        # Card input tabs
        card_tabs = Tabs(
            selected_index=0,
            tabs=[
                Tab(
                    text="⬛ Black Cards",
                    content=Container(
                        content=Column([
                            Row([
                                Text("Black cards are prompts with blanks (___)", size=11, color=Colors.GREY_400),
                                Container(expand=True),
                                self.black_count_text,
                            ]),
                            self.black_cards_area,
                        ], spacing=8),
                        padding=padding.all(10),
                    ),
                ),
                Tab(
                    text="⬜ White Cards",
                    content=Container(
                        content=Column([
                            Row([
                                Text("White cards are answers/responses", size=11, color=Colors.GREY_400),
                                Container(expand=True),
                                self.white_count_text,
                            ]),
                            self.white_cards_area,
                        ], spacing=8),
                        padding=padding.all(10),
                    ),
                ),
            ],
            expand=True,
        )
        
        return Card(
            content=Container(
                content=Column([
                    Text("🃏 Training Data", size=14, weight=FontWeight.BOLD),
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    # Theme info
                    Row([
                        self.theme_name_field,
                        self.theme_description_field,
                    ], spacing=15),
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # Import buttons
                    Row([
                        self.file_picker_btn,
                        self.load_saved_btn,
                        Container(expand=True),
                        self.total_examples_text,
                    ], spacing=10),
                    self.file_status,
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # Card input tabs
                    Container(
                        content=card_tabs,
                        height=350,
                    ),
                    
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    # Tips
                    Container(
                        content=Column([
                            Text("💡 Tips for better card training:", size=11, weight=FontWeight.W_500, color=Colors.BLUE_400),
                            Text("• Include 10-50+ black cards and 20-100+ white cards", size=10, color=Colors.GREY_400),
                            Text("• Keep a consistent theme/style across all cards", size=10, color=Colors.GREY_400),
                            Text("• Black cards should have blanks (___) for fill-in responses", size=10, color=Colors.GREY_400),
                            Text("• White cards are nouns, phrases, or funny responses", size=10, color=Colors.GREY_400),
                        ], spacing=3),
                        padding=padding.all(10),
                        bgcolor=Colors.with_opacity(0.1, Colors.BLUE_900),
                        border_radius=border_radius.all(8),
                    ),
                    
                    # Example format
                    Container(
                        content=Column([
                            Text("📝 Example Format:", size=11, weight=FontWeight.W_500),
                            Row([
                                Container(
                                    content=Column([
                                        Text("⬛ Black Card Examples:", size=10, weight=FontWeight.W_500),
                                        Text("What's a Jedi's worst nightmare?", size=9, italic=True, color=Colors.GREY_400),
                                        Text("___ is strong with the Force.", size=9, italic=True, color=Colors.GREY_400),
                                    ], spacing=3),
                                    bgcolor=Colors.GREY_900,
                                    padding=padding.all(10),
                                    border_radius=border_radius.all(6),
                                    expand=True,
                                ),
                                Container(
                                    content=Column([
                                        Text("⬜ White Card Examples:", size=10, weight=FontWeight.W_500),
                                        Text("Jar Jar Binks' family reunion", size=9, italic=True, color=Colors.GREY_400),
                                        Text("A poorly timed lightsaber ignition", size=9, italic=True, color=Colors.GREY_400),
                                    ], spacing=3),
                                    bgcolor=Colors.GREY_800,
                                    padding=padding.all(10),
                                    border_radius=border_radius.all(6),
                                    expand=True,
                                ),
                            ], spacing=10),
                        ], spacing=5),
                        padding=padding.all(10),
                        bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
                        border_radius=border_radius.all(8),
                    ),
                    
                ], spacing=10),
                padding=padding.all(15),
            ),
        )
